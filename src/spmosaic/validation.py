"""Validation helpers for spMosaic."""

from __future__ import annotations

import pandas as pd
from scipy import sparse


REQUIRED_OBS_COLUMNS = ["sample_name", "row", "col"]
REQUIRED_OBS_INDEX_NAME = "barcode"


def validate_data_type(data_type: str) -> None:
    """Validate the supported data type.

    Parameters
    ----------
    data_type : str
        Type of expression data. Must be either ``"count"`` or ``"continuous"``.

    Raises
    ------
    ValueError
        If `data_type` is not supported.
    """
    valid_types = {"count", "continuous"}
    if data_type not in valid_types:
        raise ValueError(
            f"`data_type` must be one of {valid_types}, got {data_type!r}."
        )


def validate_required_obs_columns(obs: pd.DataFrame) -> None:
    """Check that required columns exist in `.obs`.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation metadata from an AnnData object.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_OBS_COLUMNS if col not in obs.columns]
    if missing:
        raise ValueError(
            "The input .h5ad file is missing required `.obs` columns: "
            + ", ".join(missing)
        )

def validate_obs_index_name(obs: pd.DataFrame) -> None:
    """Check that the `.obs` index name is correct."""
    if obs.index.name != REQUIRED_OBS_INDEX_NAME:
        raise ValueError(
            f"The input `.obs` index name must be {REQUIRED_OBS_INDEX_NAME!r}, "
            f"got {obs.index.name!r}."
        )

def validate_sparse_matrix(x) -> None:
    """Check that `.X` is a scipy sparse matrix.

    Parameters
    ----------
    x
        Expression matrix from an AnnData object.

    Raises
    ------
    ValueError
        If `.X` is not sparse.
    """
    if not sparse.issparse(x):
        raise ValueError(
            "The input `.X` must be a scipy sparse matrix."
        )