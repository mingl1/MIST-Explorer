from core.project_naming import (
    SEGMENTATION_BASE_NAME,
    default_project_prefixed_filename,
    is_segmentation_name,
    is_temp_project_name,
    prefix_with_project_name,
)


def test_is_temp_project_name():
    assert is_temp_project_name("Temp Project 2026-01-01 00-00-00 abc123")
    assert not is_temp_project_name("My Project")
    assert not is_temp_project_name(None)


def test_prefix_with_project_name_skips_temp_projects():
    assert (
        prefix_with_project_name(
            SEGMENTATION_BASE_NAME,
            "Temp Project 2026-01-01 00-00-00 abc123",
        )
        == SEGMENTATION_BASE_NAME
    )


def test_prefix_with_project_name_adds_prefix_once():
    prefixed = prefix_with_project_name(SEGMENTATION_BASE_NAME, "KidneyPanel")
    assert prefixed == "KidneyPanel_Segmentation"
    assert prefix_with_project_name(prefixed, "KidneyPanel") == prefixed


def test_is_segmentation_name_accepts_prefixed_names():
    assert is_segmentation_name(SEGMENTATION_BASE_NAME)
    assert is_segmentation_name("KidneyPanel_Segmentation")
    assert not is_segmentation_name("KidneyPanel_Label")


def test_default_project_prefixed_filename():
    assert (
        default_project_prefixed_filename("cell_data.csv", "Kidney Panel")
        == "Kidney Panel_cell_data.csv"
    )
    assert (
        default_project_prefixed_filename(
            "cell_data.csv",
            "Temp Project 2026-01-01 00-00-00 abc123",
        )
        == "cell_data.csv"
    )
