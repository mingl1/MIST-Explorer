"""Helpers for project-aware naming of generated artifacts."""

from __future__ import annotations


SEGMENTATION_BASE_NAME = "Segmentation"
TEMP_PROJECT_PREFIX = "Temp Project "


def is_temp_project_name(project_name: str | None) -> bool:
    """Return True when the project is a temp-project variant."""
    if not project_name:
        return False
    return project_name.startswith(TEMP_PROJECT_PREFIX)


def prefix_with_project_name(
    base_name: str,
    project_name: str | None,
    *,
    is_temp_project: bool | None = None,
) -> str:
    """Prefix base_name with project_name unless this is a temp project."""
    if not project_name:
        return base_name
    temp_project = (
        is_temp_project_name(project_name)
        if is_temp_project is None
        else bool(is_temp_project)
    )
    if temp_project:
        return base_name
    prefix = f"{project_name}_"
    if base_name.startswith(prefix):
        return base_name
    return f"{prefix}{base_name}"


def is_segmentation_name(label_name: str | None) -> bool:
    """Match segmentation label names with or without a project prefix."""
    if not label_name:
        return False
    return label_name == SEGMENTATION_BASE_NAME or label_name.endswith(
        f"_{SEGMENTATION_BASE_NAME}"
    )


def _sanitize_filename_component(value: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in value)
    sanitized = sanitized.strip()
    return sanitized


def default_project_prefixed_filename(
    default_filename: str,
    project_name: str | None,
    *,
    is_temp_project: bool | None = None,
) -> str:
    """Return a default filename prefixed by project name when non-temp."""
    if not project_name:
        return default_filename

    temp_project = (
        is_temp_project_name(project_name)
        if is_temp_project is None
        else bool(is_temp_project)
    )
    if temp_project:
        return default_filename

    safe_project_name = _sanitize_filename_component(project_name)
    if not safe_project_name:
        return default_filename
    prefix = f"{safe_project_name}_"
    if default_filename.startswith(prefix):
        return default_filename
    return f"{prefix}{default_filename}"
