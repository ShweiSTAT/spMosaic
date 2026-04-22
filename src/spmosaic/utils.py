"""Utility helpers for spMosaic."""

from __future__ import annotations

import os
import shutil
import subprocess
import random

import torch 
from pathlib import Path

import numpy as np


def get_rscript_path(rscript_path: str | Path | None = None) -> str:
    """Resolve which Rscript executable to use.

    Resolution priority is:

    1. the explicit ``rscript_path`` argument
    2. the environment variable ``SPMOSAIC_RSCRIPT``
    3. the first ``Rscript`` found on ``PATH``

    Parameters
    ----------
    rscript_path : str or Path or None
        Optional explicit path to the Rscript executable.

    Returns
    -------
    str
        Full path to the Rscript executable.

    Raises
    ------
    RuntimeError
        If no usable Rscript executable can be found.
    """
    if rscript_path is not None:
        path = Path(rscript_path)
        if not path.exists():
            raise RuntimeError(
                f"Provided `rscript_path` does not exist: {path}"
            )
        return str(path)

    env_rscript = os.environ.get("SPMOSAIC_RSCRIPT")
    if env_rscript:
        path = Path(env_rscript)
        if not path.exists():
            raise RuntimeError(
                "Environment variable `SPMOSAIC_RSCRIPT` points to a missing file: "
                f"{path}"
            )
        return str(path)

    detected = shutil.which("Rscript")
    if detected is None:
        raise RuntimeError(
            "Rscript not found. Provide `rscript_path`, set the environment "
            "variable `SPMOSAIC_RSCRIPT`, or activate an environment that "
            "contains Rscript."
        )
    return detected


def get_r_version(rscript_path: str | Path | None = None) -> str:
    """Return the version string of the R installation being used.

    Parameters
    ----------
    rscript_path : str or Path or None
        Optional explicit path to the Rscript executable.

    Returns
    -------
    str
        Version information reported by Rscript.
    """
    rscript = get_rscript_path(rscript_path=rscript_path)
    completed = subprocess.run(
        [rscript, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def set_seed(seed: int = 888, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default=888
        Random seed.
    deterministic : bool, default=True
        Whether to request deterministic torch behavior where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass