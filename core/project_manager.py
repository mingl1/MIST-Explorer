import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import tifffile

from models.workspace import ImageMetadata, ProjectMetadata

logger = logging.getLogger(__name__)


class ProjectManager:
    @staticmethod
    def get_storage_path() -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "MIST-Explorer"
        elif system == "Linux":
            return Path.home() / ".local" / "share" / "MIST-Explorer"
        elif system == "Windows":
            return Path.home() / "AppData" / "Roaming" / "MIST-Explorer"
        else:
            return Path.home() / "MIST-Explorer"

    @staticmethod
    def get_projects_path() -> Path:
        return ProjectManager.get_storage_path() / "projects"

    @staticmethod
    def ensure_storage_exists():
        projects_path = ProjectManager.get_projects_path()
        projects_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_project(
        name: str, base_path: Optional[Path] = None, is_temporary: bool = False
    ) -> Path:
        ProjectManager.ensure_storage_exists()

        if base_path is None:
            base_path = ProjectManager.get_projects_path()

        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        project_folder = base_path / f"{safe_name}.mist"
        project_folder.mkdir(parents=True, exist_ok=True)

        (project_folder / "images").mkdir(exist_ok=True)
        (project_folder / "thumbnails").mkdir(exist_ok=True)

        metadata = ProjectMetadata(
            name=name,
            path=project_folder,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            is_temporary=is_temporary,
        )
        ProjectManager._save_metadata(metadata)

        return project_folder

    @staticmethod
    def create_temp_project(base_path: Optional[Path] = None) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        suffix = uuid.uuid4().hex[:6]
        return ProjectManager.create_project(
            f"Temp Project {timestamp} {suffix}",
            base_path=base_path,
            is_temporary=True,
        )

    @staticmethod
    def _is_temp_project(metadata: ProjectMetadata) -> bool:
        if metadata.is_temporary:
            return True
        # Backward-compat: hide temporary projects created before is_temporary existed.
        name = metadata.name.strip().lower()
        folder_stem = metadata.path.stem.strip().lower()
        return name.startswith("temp project") or folder_stem.startswith("temp project")

    @staticmethod
    def _save_metadata(metadata: ProjectMetadata):
        metadata_path = metadata.path / "project.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    @staticmethod
    def load_metadata(project_path: Path) -> Optional[ProjectMetadata]:
        metadata_path = project_path / "project.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return ProjectMetadata.from_dict(data)
        except Exception as e:
            logger.error("Failed to load project metadata from '%s': %s", metadata_path, e, exc_info=True)
            return None

    @staticmethod
    def load_project(project_path: Path) -> Optional[ProjectMetadata]:
        return ProjectManager.load_metadata(project_path)

    @staticmethod
    def get_recent_projects() -> List[ProjectMetadata]:
        ProjectManager.ensure_storage_exists()
        projects = []

        projects_path = ProjectManager.get_projects_path()
        if not projects_path.exists():
            return []

        for folder in projects_path.iterdir():
            if folder.is_dir() and folder.suffix == ".mist":
                metadata = ProjectManager.load_metadata(folder)
                if metadata is not None:
                    if ProjectManager._is_temp_project(metadata):
                        continue
                    projects.append(metadata)

        return sorted(projects, key=lambda p: p.modified_at, reverse=True)

    @staticmethod
    def save_image(
        project_path: Path,
        image_uuid: str,
        channel_data: Dict[str, "ImageWrapper"],
        image_name: str,
        channel_count: int,
        original_filename: str = "",
        contrast_settings: Optional[Dict[str, Tuple[int, int]]] = None,
        channel_display_names: Optional[Dict[str, str]] = None,
    ) -> ImageMetadata:
        image_folder = project_path / "images" / image_uuid
        image_folder.mkdir(parents=True, exist_ok=True)

        for channel_name, wrapper in channel_data.items():
            channel_num = channel_name.replace("Channel ", "")
            channel_path = image_folder / f"channel_{channel_num}.tif"
            try:
                tifffile.imwrite(
                    str(channel_path),
                    wrapper.data,
                    photometric="minisblack",
                    imagej=True,
                )
            except Exception as e:
                logger.error("Failed to save channel '%s' to '%s': %s", channel_name, channel_path, e, exc_info=True)
                raise

        if contrast_settings is None:
            contrast_settings = {}
        if channel_display_names is None:
            channel_display_names = {}

        image_metadata = ImageMetadata(
            uuid=image_uuid,
            name=image_name,
            channel_count=channel_count,
            original_filename=original_filename,
            contrast_settings=contrast_settings,
            channel_display_names=channel_display_names,
        )

        metadata = ProjectManager.load_metadata(project_path)
        if metadata is not None:
            existing = metadata.get_image(image_uuid)
            if existing is None:
                metadata.add_image(image_metadata)
            else:
                metadata.update_image(image_metadata)
            ProjectManager._save_metadata(metadata)

        return image_metadata

    @staticmethod
    def load_image(project_path: Path, image_uuid: str, channel: str = "Channel 1"):
        image_folder = project_path / "images" / image_uuid
        channel_num = channel.replace("Channel ", "")
        channel_path = image_folder / f"channel_{channel_num}.tif"

        if channel_path.exists():
            try:
                return tifffile.imread(str(channel_path))
            except Exception as e:
                logger.error("Failed to load image from '%s': %s", channel_path, e, exc_info=True)
                return None
        return None

    @staticmethod
    def list_saved_channels(project_path: Path, image_uuid: str) -> List[str]:
        image_folder = project_path / "images" / image_uuid
        if not image_folder.exists():
            return []

        channels = []
        for channel_path in image_folder.glob("channel_*.tif"):
            suffix = channel_path.stem[len("channel_") :]
            try:
                channel_num = int(suffix)
            except ValueError:
                continue
            channels.append(channel_num)

        channels.sort()
        return [f"Channel {channel_num}" for channel_num in channels]

    @staticmethod
    def get_image_metadata(
        project_path: Path, image_uuid: str
    ) -> Optional[ImageMetadata]:
        metadata = ProjectManager.load_metadata(project_path)
        if metadata is not None:
            return metadata.get_image(image_uuid)
        return None

    @staticmethod
    def update_image_contrast(
        project_path: Path,
        image_uuid: str,
        channel: str,
        min_val: int,
        max_val: int,
    ):
        metadata = ProjectManager.load_metadata(project_path)
        if metadata is not None:
            image = metadata.get_image(image_uuid)
            if image is not None:
                image.contrast_settings[channel] = (min_val, max_val)
                metadata.update_image(image)
                ProjectManager._save_metadata(metadata)

    @staticmethod
    def delete_image(project_path: Path, image_uuid: str) -> bool:
        """Delete an image from the project (files and metadata)."""
        import shutil

        # Delete image files from disk
        image_folder = project_path / "images" / image_uuid
        if image_folder.exists():
            shutil.rmtree(image_folder)

        # Delete thumbnail
        thumbnail_path = project_path / "thumbnails" / f"{image_uuid}.png"
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        # Update metadata
        metadata = ProjectManager.load_metadata(project_path)
        if metadata is not None:
            metadata.remove_image(image_uuid)
            ProjectManager._save_metadata(metadata)
            return True
        return False

    @staticmethod
    def delete_project(project_path: Path) -> bool:
        import shutil

        if project_path.exists() and project_path.suffix == ".mist":
            shutil.rmtree(project_path)
            return True
        return False

    @staticmethod
    def open_project_folder(project_path: Path):
        import subprocess

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(["open", str(project_path)])
            elif system == "Linux":
                subprocess.run(["xdg-open", str(project_path)])
            elif system == "Windows":
                subprocess.run(["explorer", str(project_path)])
        except Exception as e:
            logger.error("Failed to open project folder '%s': %s", project_path, e, exc_info=True)

    @staticmethod
    def get_folder_size(path: Path) -> int:
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    @staticmethod
    def format_size(size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
