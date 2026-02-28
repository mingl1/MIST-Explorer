from pathlib import Path

import numpy as np
import tifffile

from core.project_manager import ProjectManager


def test_list_saved_channels_handles_non_sequential_files(tmp_path: Path):
    project_path = tmp_path / "sample.mist"
    image_uuid = "image-123"
    image_folder = project_path / "images" / image_uuid
    image_folder.mkdir(parents=True)

    arr = np.zeros((4, 4), dtype=np.uint16)
    tifffile.imwrite(str(image_folder / "channel_3.tif"), arr)
    tifffile.imwrite(str(image_folder / "channel_1.tif"), arr)
    tifffile.imwrite(str(image_folder / "channel_bad.tif"), arr)

    channels = ProjectManager.list_saved_channels(project_path, image_uuid)

    assert channels == ["Channel 1", "Channel 3"]


def test_create_temp_project_builds_valid_project_folder(tmp_path: Path):
    project_path = ProjectManager.create_temp_project(base_path=tmp_path)

    assert project_path.exists()
    assert project_path.suffix == ".mist"
    assert (project_path / "project.json").exists()
    assert (project_path / "images").exists()
    assert (project_path / "thumbnails").exists()


def test_temp_project_not_listed_in_recent_projects(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ProjectManager, "get_projects_path", staticmethod(lambda: tmp_path)
    )

    normal_project = ProjectManager.create_project("Regular Project", base_path=tmp_path)
    temp_project = ProjectManager.create_temp_project(base_path=tmp_path)

    recent_paths = {p.path for p in ProjectManager.get_recent_projects()}

    assert normal_project in recent_paths
    assert temp_project not in recent_paths


def test_legacy_temp_named_project_not_listed_in_recent_projects(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        ProjectManager, "get_projects_path", staticmethod(lambda: tmp_path)
    )

    legacy_temp = ProjectManager.create_project(
        "Temp Project Legacy No Flag", base_path=tmp_path, is_temporary=False
    )
    regular = ProjectManager.create_project("Visible Project", base_path=tmp_path)

    recent_paths = {p.path for p in ProjectManager.get_recent_projects()}

    assert regular in recent_paths
    assert legacy_temp not in recent_paths
