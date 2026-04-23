"""Stage 1 gene smoothing functions for spMosaic."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
import importlib.resources as ir

from .io import export_h5ad_for_r
from .utils import get_rscript_path


def run_gam_rscript(
    input_root: str | Path,
    prefix: str,
    ncores: int,
    k_num: int,
    if_fix_this: bool,
    data_type: str,
    out_root: str | Path,
    rscript_path: str | Path | None = None,
    blas_threads: int | None = 1,
) -> dict:
    """Run the packaged R GAM smoothing script.

    This function locates the packaged R script, builds the command-line
    arguments, executes the script using ``Rscript``, and returns the main
    output file paths.

    Parameters
    ----------
    input_root : str or Path
        Root directory containing the ``{prefix}_gam_input`` folder.
    prefix : str
        Prefix used in input and output file naming.
    ncores : int
        Number of CPU cores to use in the R smoothing step.
    k_num : int
        Number of spline basis functions used by the GAM.
    if_fix_this : bool
        Whether to fix the spline basis degrees of freedom in the GAM.
    data_type : str
        Type of input expression data. Must be either ``"count"`` or
        ``"continuous"``.
    out_root : str or Path
        Root directory where ``{prefix}_gam_output`` will be created.
    rscript_path : str or Path or None, default=None
        Optional explicit path to the Rscript executable.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries in the R subprocess. Setting this to 1 helps avoid thread
        oversubscription on HPC systems, especially when higher-level
        parallelism is already controlled by ``ncores``. If None, the existing
        environment settings are passed through unchanged.

    Returns
    -------
    dict
        Dictionary containing the resolved Rscript path, command, console
        output, BLAS thread setting, and paths to the expected stage 1 output
        files.

    Raises
    ------
    RuntimeError
        If the R script execution fails.
    """
    input_root = Path(input_root)
    out_root = Path(out_root)

    rscript = get_rscript_path(rscript_path=rscript_path)

    r_resource = ir.files("spmosaic").joinpath("r/fit_gam.R")
    with ir.as_file(r_resource) as r_script_path:
        cmd = [
            rscript,
            str(r_script_path),
            str(input_root),
            prefix,
            str(ncores),
            str(k_num),
            "True" if if_fix_this else "False",
            data_type,
            str(out_root),
        ]

        print("Using Rscript:", rscript)
        print("Running command:")
        print(" ".join(cmd))

        env = os.environ.copy()
        if blas_threads is not None:
            env["OMP_NUM_THREADS"] = str(blas_threads)
            env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
            env["MKL_NUM_THREADS"] = str(blas_threads)
            env["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
            env["BLAS_NUM_THREADS"] = str(blas_threads)

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                text=True,
                env=env,
                # capture_output=True,  # uncomment if you want to capture R output instead of printing live
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "R GAM script failed.\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"STDOUT:\n{e.stdout}\n\n"
                f"STDERR:\n{e.stderr}"
            ) from e

    output_dir = out_root / f"{prefix}_gam_output"

    return {
        "rscript": rscript,
        "command": cmd,
        "blas_threads": blas_threads,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "smoothed_path": str(output_dir / f"{prefix}_smoothed_gene_exp.csv"),
        "raw_stat_path": str(output_dir / f"{prefix}_GAM_RawStat.csv"),
        "combined_stat_path": str(output_dir / f"{prefix}_GAM_CombinedStat.csv"),
        "spots_metadata_path": str(output_dir / f"{prefix}_spots_metadata.csv"),
    }


def gene_smooth(
    h5ad_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    data_type: str,
    ncores: int = 1,
    k_num: int = 40,
    if_fix_this: bool = True,
    rscript_path: str | Path | None = None,
    blas_threads: int | None = 1,
) -> dict:
    """Run stage 1 gene smoothing for spMosaic.

    This function performs the complete stage 1 workflow:

    1. validates and exports the input ``.h5ad`` file into intermediate files
       for R
    2. runs the packaged R GAM smoothing script
    3. returns paths to the intermediate and output files

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the input ``.h5ad`` file. The file is expected to contain all
        spots from all samples, with required observation columns including
        ``sample_name``, ``row``, and ``col``.
    out_dir : str or Path
        Root output directory for intermediate files and GAM results.
    prefix : str
        Prefix used to name intermediate and output files for this analysis.
    data_type : str
        Type of input expression data. Must be either ``"count"`` or
        ``"continuous"``.
    ncores : int, default=1
        Number of CPU cores to use during the R GAM smoothing step.
    k_num : int, default=40
        Number of spline basis functions used by the GAM.
    if_fix_this : bool, default=True
        Whether to fix the spline basis degrees of freedom in the GAM.
    rscript_path : str or Path or None, default=None
        Optional explicit path to the Rscript executable. If not provided, the
        function will try ``SPMOSAIC_RSCRIPT`` and then ``Rscript`` on PATH.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries in the R subprocess. Setting this to 1 helps avoid thread
        oversubscription on HPC systems, especially when ``ncores`` is greater
        than 1 or when hidden multithreading from numerical libraries causes
        unexpected slowdown. If None, the existing environment settings are
        left unchanged.

    Returns
    -------
    dict
        Dictionary summarizing stage 1 results, including:

        - exported intermediate file paths
        - GAM output file paths
        - the Rscript executable used
        - the BLAS/OpenMP thread setting used for the R subprocess
        - the R console stdout/stderr

    Raises
    ------
    ValueError
        If invalid input arguments are passed.
    RuntimeError
        If the Rscript executable cannot be found or the R GAM script fails.
    """
    if ncores < 1:
        raise ValueError("`ncores` must be at least 1.")

    if k_num < 1:
        raise ValueError("`k_num` must be at least 1.")

    if blas_threads is not None and blas_threads < 1:
        raise ValueError("`blas_threads` must be at least 1 or None.")

    out_dir = Path(out_dir)

    print("Step 1-1: Exporting .h5ad intermediate files for R GAM smoothing...")
    export_result = export_h5ad_for_r(
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        prefix=prefix,
        data_type=data_type,
    )

    print("Step 1-2: Running R GAM smoothing script...")
    gam_result = run_gam_rscript(
        input_root=out_dir,
        prefix=prefix,
        ncores=ncores,
        k_num=k_num,
        if_fix_this=if_fix_this,
        data_type=data_type,
        out_root=out_dir,
        rscript_path=rscript_path,
        blas_threads=blas_threads,
    )
    print("All done for stage 1!")

    return {
        "stage": "gene_smooth",
        "prefix": prefix,
        "data_type": data_type,
        "ncores": ncores,
        "k_num": k_num,
        "if_fix_this": if_fix_this,
        "blas_threads": blas_threads,
        "export": export_result,
        "gam": gam_result,
    }