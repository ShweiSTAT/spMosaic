"""Input/output helpers for spMosaic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmwrite

from .validation import (
    validate_data_type,
    validate_required_obs_columns,
    validate_obs_index_name,
    validate_sparse_matrix,
)


def export_h5ad_for_r(
    h5ad_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    data_type: str,
) -> dict:
    """Export an input .h5ad file into intermediate files for R GAM smoothing.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the input .h5ad file. The file must contain gene expression in
        `.X` and observation metadata in `.obs`, including the columns
        `sample_name`, `row`, and `col` and the index "barcode".
    out_dir : str or Path
        Output directory where the intermediate files will be written.
    prefix : str
        Prefix used for naming output files, for example ``"Donor1"``.
    data_type : str
        Type of expression data. Must be either ``"count"`` or ``"continuous"``.

    Returns
    -------
    dict
        Dictionary containing the output directory and exported file paths.

    Raises
    ------
    ValueError
        If required metadata columns are missing, if `data_type` is invalid,
        or if the expression matrix is not sparse.
    """
    h5ad_path = Path(h5ad_path)
    out_dir = Path(out_dir)
    output_dir = out_dir / f"{prefix}_gam_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AnnData from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    validate_data_type(data_type)
    validate_required_obs_columns(adata.obs)
    validate_obs_index_name(adata.obs)
    validate_sparse_matrix(adata.X)

    if data_type == "count" and "sum_umi" not in adata.obs.columns:
        adata.obs["sum_umi"] = np.asarray(adata.X.sum(axis=1)).ravel()

    mtx_path = output_dir / f"{prefix}_counts.mtx"
    genes_path = output_dir / f"{prefix}_genes.tsv"
    barcodes_path = output_dir / f"{prefix}_barcodes.tsv"
    obs_path = output_dir / f"{prefix}_obs_metadata.csv"

    print("Saving expression matrix...")
    mmwrite(str(mtx_path), adata.X)

    print("Saving gene names...")
    pd.Series(adata.var_names).to_csv(genes_path, index=False, header=False)

    print("Saving barcodes...")
    pd.Series(adata.obs_names).to_csv(barcodes_path, index=False, header=False)

    print("Saving observation metadata...")
    adata.obs.to_csv(obs_path)

    print(f"Export complete. Intermediate files saved to {output_dir}")

    return {
        "stage": "stage 0: preparing for gene smooth by GAM",
        "output_dir": str(output_dir),
        "mtx_path": str(mtx_path),
        "genes_path": str(genes_path),
        "barcodes_path": str(barcodes_path),
        "obs_path": str(obs_path),
    }