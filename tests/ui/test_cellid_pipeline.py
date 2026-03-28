import numpy as np
import pandas as pd

from core.cell_intensity import (
    CellIntensity,
    build_channel_cell_dataframe,
    merge_channel_cell_data,
)
from ui.view_tab import collapse_duplicate_cell_ids


def test_build_channel_cell_dataframe_includes_unique_integer_cellid():
    median_values = {1: [10.0, 20.0], 2: [30.0, 40.0], 3: [50.0, 60.0]}
    centroids = {1: (11, 101), 2: (22, 202), 3: (33, 303)}

    df = build_channel_cell_dataframe(
        median_values_for_cell_data_dict=median_values,
        cell_centroids=centroids,
        protein_headers=["P1", "P2"],
    )

    assert "CellID" in df.columns
    assert df["CellID"].dtype.kind in {"i", "u"}
    assert df["CellID"].is_unique
    assert df.columns.tolist() == ["CellID", "Global X", "Global Y", "P1", "P2"]
    assert len(df) == 3


def test_merge_channel_cell_data_uses_cellid_and_preserves_first_channel_coords():
    first = pd.DataFrame(
        {
            "CellID": [1, 2],
            "Global X": [10, 20],
            "Global Y": [100, 200],
            "CD3": [1.0, 2.0],
        }
    )
    second = pd.DataFrame(
        {
            "CellID": [1, 2],
            "Global X": [999, 999],
            "Global Y": [999, 999],
            "CD3": [3.0, 4.0],
            "CD4": [5.0, 6.0],
        }
    )

    merged = merge_channel_cell_data(
        existing_df=first,
        curr_cell_data=second,
        first_channel_num=1,
        channel_num=2,
    )

    assert merged.columns.tolist() == [
        "CellID",
        "Global X",
        "Global Y",
        "CD3_cy1",
        "CD3_cy2",
        "CD4",
    ]
    assert merged["Global X"].tolist() == [10, 20]
    assert merged["Global Y"].tolist() == [100, 200]
    assert len(merged) == 2


def test_collapse_duplicate_cell_ids_aggregates_numeric_and_keeps_first_coords():
    df = pd.DataFrame(
        {
            "CellID": [1, 1, 2],
            "Global X": [100, 999, 200],
            "Global Y": [300, 888, 400],
            "ProteinA": [10.0, 30.0, 40.0],
            "Cluster": ["A", "B", "C"],
        }
    )

    collapsed = collapse_duplicate_cell_ids(df)
    row_1 = collapsed[collapsed["CellID"] == 1].iloc[0]

    assert len(collapsed) == 2
    assert row_1["Global X"] == 100
    assert row_1["Global Y"] == 300
    assert row_1["ProteinA"] == 20.0
    assert row_1["Cluster"] == "A"


def test_duplicate_collapse_prevents_lut_assignment_mismatch():
    df = pd.DataFrame(
        {
            "CellID": [1, 1, 2, 3],
            "Global X": [0, 1, 2, 3],
            "Global Y": [0, 1, 2, 3],
            "ProteinA": [10.0, 30.0, 40.0, 50.0],
        }
    )
    collapsed = collapse_duplicate_cell_ids(df)
    indexed = collapsed.set_index("CellID")

    protein_series = indexed["ProteinA"]
    max_id_in_image = 3
    lut_data = np.zeros(max_id_in_image + 1)
    valid_indices = protein_series.index[protein_series.index <= max_id_in_image]
    values = protein_series.loc[valid_indices].values

    # Regression: these lengths must match to avoid numpy broadcast errors.
    assert len(valid_indices) == len(values)
    lut_data[valid_indices] = values
    assert lut_data.tolist() == [0.0, 20.0, 40.0, 50.0]


def test_compute_all_centroids_supports_cancel():
    worker = CellIntensity()
    worker.segmentation_labels = np.array([[0, 1], [1, 1]], dtype=np.uint16)

    worker.cancel()
    centroids = worker.compute_all_centroids()

    assert centroids is None
