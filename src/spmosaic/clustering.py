# """Clustering utilities for spMosaic."""


# from __future__ import annotations

# import torch 

# import numpy as np
# import pandas as pd


# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture

# import subprocess
# import tempfile
# from pathlib import Path
# import importlib.resources as ir

# from .utils import get_rscript_path

# def run_mclust_subprocess(
#     embedding: np.ndarray,
#     num_clusters: int,
#     random_state: int,
#     model_name: str = "EEE",
#     rscript_path: str | Path | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Run R `mclust` in a subprocess and return cluster labels and centers.

#     This function writes a latent embedding matrix to a temporary CSV file,
#     calls a packaged R script that runs `mclust::Mclust`, and reads the
#     resulting cluster labels and cluster centers back into Python.

#     Parameters
#     ----------
#     embedding : np.ndarray
#         Two-dimensional latent embedding matrix of shape
#         ``(n_samples, n_features)``.
#     num_clusters : int
#         Number of clusters to fit.
#     random_state : int
#         Random seed passed to the R script for reproducibility.
#     model_name : str, default="EEE"
#         Covariance model name used by R `mclust`. Examples include
#         ``"EEE"``, ``"VVV"``, and other valid `mclust` model names.
#     rscript_path : str or Path or None, default=None
#         Optional explicit path to the `Rscript` executable. If not provided,
#         the function will attempt to locate `Rscript` from the environment.

#     Returns
#     -------
#     labels : np.ndarray
#         One-dimensional array of predicted cluster labels of shape
#         ``(n_samples,)``.
#     centers : np.ndarray
#         Two-dimensional array of cluster centers of shape
#         ``(num_clusters, n_features)``.

#     Raises
#     ------
#     ValueError
#         If `embedding` is not two-dimensional, or if it contains NaN or Inf.
#     RuntimeError
#         If `Rscript` cannot be found.
#     subprocess.CalledProcessError
#         If the R subprocess exits with an error.

#     Notes
#     -----
#     This function uses a temporary directory to store intermediate CSV files.
#     These files are automatically removed after the subprocess finishes.

#     The actual clustering is performed by the packaged R script
#     ``r/run_mclust_init.R``.
#     """
#     if embedding.ndim != 2:
#         raise ValueError("`embedding` must be 2D.")

#     if np.isnan(embedding).any():
#         raise ValueError("`embedding` contains NaN.")
#     if np.isinf(embedding).any():
#         raise ValueError("`embedding` contains Inf.")

#     rscript = get_rscript_path(rscript_path)

#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmpdir = Path(tmpdir)

#         embedding_csv = tmpdir / "embedding.csv"
#         labels_csv = tmpdir / "labels.csv"
#         centers_csv = tmpdir / "centers.csv"

#         pd.DataFrame(embedding).to_csv(embedding_csv, index=False)

#         r_resource = ir.files("spmosaic").joinpath("r/run_mclust_init.R")
#         with ir.as_file(r_resource) as r_script_path:
#             cmd = [
#                 rscript,
#                 str(r_script_path),
#                 str(embedding_csv),
#                 str(labels_csv),
#                 str(centers_csv),
#                 str(num_clusters),
#                 model_name,
#                 str(random_state),
#             ]

#             print("Using Rscript:", rscript)
#             print("Running command:")
#             print(" ".join(cmd))

#             subprocess.run(cmd, check=True, text=True)

#         labels = pd.read_csv(labels_csv)["cluster"].to_numpy(dtype=int)
#         centers = pd.read_csv(centers_csv).to_numpy(dtype=float)

#     return labels, centers


# def initialize_clusters(
#     encoder_G,
#     data,
#     num_clusters,
#     method="gmm",
#     seeds=888,
# ):
#     """Initialize cluster labels and cluster centers for DEC refinement.

#     This function first computes the biological embedding by passing the input
#     data through the pretrained biological encoder ``encoder_G``. It then
#     applies a clustering method to the embedding in order to obtain initial
#     cluster assignments and cluster centers for DEC training.

#     Supported clustering methods are:

#     - ``"gmm"``: Gaussian mixture model clustering
#     - ``"mclust"``: R `mclust` clustering through a subprocess
#     - ``"kmeans"``: K-means clustering

#     Parameters
#     ----------
#     encoder_G : nn.Module
#         Pretrained biological encoder from the autoencoder.
#     data : torch.Tensor or np.ndarray
#         Input data matrix of shape ``(n_samples, input_dim)``.
#     num_clusters : int
#         Number of clusters.
#     method : str, default="gmm"
#         Clustering method used for initialization. Must be one of
#         ``"gmm"``, ``"mclust"``, or ``"kmeans"``.
#     seeds : int, default=888
#         Random seed used for clustering initialization.

#     Returns
#     -------
#     cluster_centers : torch.Tensor
#         Initialized cluster centers of shape
#         ``(num_clusters, embedding_dim)`` on the same device as
#         ``encoder_G``.
#     y_pred : np.ndarray
#         Predicted cluster labels of shape ``(n_samples,)``.

#     Raises
#     ------
#     TypeError
#         If `data` is neither a NumPy array nor a torch tensor.
#     ValueError
#         If `method` is not one of the supported clustering methods.

#     Notes
#     -----
#     The biological embedding is computed with the encoder in evaluation mode
#     and without gradient tracking.

#     For the ``"mclust"`` option, clustering is performed by an external R
#     subprocess and the results are read back into Python.
#     """
#     encoder_G.eval()

#     encoder_device = next(encoder_G.parameters()).device

#     if isinstance(data, np.ndarray):
#         data_tensor = torch.tensor(data, dtype=torch.float32, device=encoder_device)
#     elif isinstance(data, torch.Tensor):
#         data_tensor = data.to(encoder_device, dtype=torch.float32)
#     else:
#         raise TypeError("`data` must be a numpy array or torch tensor.")

#     with torch.no_grad():
#         h_G = encoder_G(data_tensor).detach().cpu().numpy()

#     print(f"Initializing clusters using {method.upper()}...")

#     if method == "gmm":
#         gmm = GaussianMixture(
#             n_components=num_clusters,
#             covariance_type="full",
#             random_state=seeds
#         )
#         gmm.fit(h_G)
#         y_pred = gmm.predict(h_G)
#         cluster_centers = torch.tensor(
#             gmm.means_,
#             dtype=torch.float32,
#             device=encoder_device
#         )

#     elif method == "mclust":
#         y_pred, cluster_centers_np = run_mclust_subprocess(
#             embedding=h_G,
#             num_clusters=num_clusters,
#             model_name="EEE",
#             random_state=seeds
#         )
#         cluster_centers = torch.tensor(
#             cluster_centers_np,
#             dtype=torch.float32,
#             device=encoder_device
#         )

#     elif method == "kmeans":
#         kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=seeds)
#         kmeans.fit(h_G)
#         y_pred = kmeans.labels_

#         cluster_centers_np = np.array([
#             h_G[y_pred == cluster_id].mean(axis=0)
#             for cluster_id in np.unique(y_pred)
#         ])
#         cluster_centers = torch.tensor(
#             cluster_centers_np,
#             dtype=torch.float32,
#             device=encoder_device
#         )

#     else:
#         raise ValueError(f"Unknown clustering method: {method}")

#     return cluster_centers, y_pred

"""Clustering utilities for spMosaic."""

from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import subprocess
import tempfile
from pathlib import Path
import importlib.resources as ir

from .utils import get_rscript_path


def run_mclust_subprocess(
    embedding: np.ndarray,
    num_clusters: int,
    random_state: int,
    model_name: str = "EEE",
    rscript_path: str | Path | None = None,
    blas_threads: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run R ``mclust`` in a subprocess and return cluster labels and centers.

    This function writes a latent embedding matrix to a temporary CSV file,
    calls a packaged R script that runs ``mclust::Mclust``, and reads the
    resulting cluster labels and cluster centers back into Python.

    Parameters
    ----------
    embedding : np.ndarray
        Two-dimensional latent embedding matrix of shape
        ``(n_samples, n_features)``.
    num_clusters : int
        Number of clusters to fit.
    random_state : int
        Random seed passed to the R script for reproducibility.
    model_name : str, default="EEE"
        Covariance model name used by R ``mclust``. Examples include
        ``"EEE"``, ``"VVV"``, and other valid ``mclust`` model names.
    rscript_path : str or Path or None, default=None
        Optional explicit path to the ``Rscript`` executable. If not provided,
        the function will attempt to locate ``Rscript`` from the environment.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries in the R subprocess. Setting this to 1 helps avoid thread
        oversubscription on HPC systems. If None, the existing environment
        settings are passed through unchanged.

    Returns
    -------
    labels : np.ndarray
        One-dimensional array of predicted cluster labels of shape
        ``(n_samples,)``.
    centers : np.ndarray
        Two-dimensional array of cluster centers of shape
        ``(num_clusters, n_features)``.

    Raises
    ------
    ValueError
        If ``embedding`` is not two-dimensional, if it contains NaN or Inf,
        or if ``blas_threads`` is less than 1 when not None.
    RuntimeError
        If ``Rscript`` cannot be found.
    subprocess.CalledProcessError
        If the R subprocess exits with an error.

    Notes
    -----
    This function uses a temporary directory to store intermediate CSV files.
    These files are automatically removed after the subprocess finishes.

    The actual clustering is performed by the packaged R script
    ``r/run_mclust_init.R``.
    """
    if embedding.ndim != 2:
        raise ValueError("`embedding` must be 2D.")

    if np.isnan(embedding).any():
        raise ValueError("`embedding` contains NaN.")
    if np.isinf(embedding).any():
        raise ValueError("`embedding` contains Inf.")

    if blas_threads is not None and blas_threads < 1:
        raise ValueError("`blas_threads` must be at least 1 or None.")

    rscript = get_rscript_path(rscript_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        embedding_csv = tmpdir / "embedding.csv"
        labels_csv = tmpdir / "labels.csv"
        centers_csv = tmpdir / "centers.csv"

        pd.DataFrame(embedding).to_csv(embedding_csv, index=False)

        r_resource = ir.files("spmosaic").joinpath("r/run_mclust_init.R")
        with ir.as_file(r_resource) as r_script_path:
            cmd = [
                rscript,
                str(r_script_path),
                str(embedding_csv),
                str(labels_csv),
                str(centers_csv),
                str(num_clusters),
                model_name,
                str(random_state),
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

            subprocess.run(cmd, check=True, text=True, env=env)

        labels = pd.read_csv(labels_csv)["cluster"].to_numpy(dtype=int)
        centers = pd.read_csv(centers_csv).to_numpy(dtype=float)

    return labels, centers


def initialize_clusters(
    encoder_G,
    data,
    num_clusters,
    method="gmm",
    seeds=888,
    blas_threads: int | None = 1,
):
    """Initialize cluster labels and cluster centers for DEC refinement.

    This function first computes the biological embedding by passing the input
    data through the pretrained biological encoder ``encoder_G``. It then
    applies a clustering method to the embedding in order to obtain initial
    cluster assignments and cluster centers for DEC training.

    Supported clustering methods are:

    - ``"gmm"``: Gaussian mixture model clustering
    - ``"mclust"``: R ``mclust`` clustering through a subprocess
    - ``"kmeans"``: K-means clustering

    Parameters
    ----------
    encoder_G : nn.Module
        Pretrained biological encoder from the autoencoder.
    data : torch.Tensor or np.ndarray
        Input data matrix of shape ``(n_samples, input_dim)``.
    num_clusters : int
        Number of clusters.
    method : str, default="gmm"
        Clustering method used for initialization. Must be one of
        ``"gmm"``, ``"mclust"``, or ``"kmeans"``.
    seeds : int, default=888
        Random seed used for clustering initialization.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries when ``method="mclust"`` is used through an R subprocess.
        Setting this to 1 helps avoid hidden multithreading and improves
        stability on HPC systems. Ignored for ``"gmm"`` and ``"kmeans"``.
        If None, the existing environment settings are left unchanged.

    Returns
    -------
    cluster_centers : torch.Tensor
        Initialized cluster centers of shape
        ``(num_clusters, embedding_dim)`` on the same device as
        ``encoder_G``.
    y_pred : np.ndarray
        Predicted cluster labels of shape ``(n_samples,)``.

    Raises
    ------
    TypeError
        If ``data`` is neither a NumPy array nor a torch tensor.
    ValueError
        If ``method`` is not one of the supported clustering methods, or if
        ``blas_threads`` is less than 1 when not None.

    Notes
    -----
    The biological embedding is computed with the encoder in evaluation mode
    and without gradient tracking.

    For the ``"mclust"`` option, clustering is performed by an external R
    subprocess and the results are read back into Python.
    """
    if blas_threads is not None and blas_threads < 1:
        raise ValueError("`blas_threads` must be at least 1 or None.")

    encoder_G.eval()

    encoder_device = next(encoder_G.parameters()).device

    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32, device=encoder_device)
    elif isinstance(data, torch.Tensor):
        data_tensor = data.to(encoder_device, dtype=torch.float32)
    else:
        raise TypeError("`data` must be a numpy array or torch tensor.")

    with torch.no_grad():
        h_G = encoder_G(data_tensor).detach().cpu().numpy()

    print(f"Initializing clusters using {method.upper()}...")

    if method == "gmm":
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type="full",
            random_state=seeds
        )
        gmm.fit(h_G)
        y_pred = gmm.predict(h_G)
        cluster_centers = torch.tensor(
            gmm.means_,
            dtype=torch.float32,
            device=encoder_device
        )

    elif method == "mclust":
        y_pred, cluster_centers_np = run_mclust_subprocess(
            embedding=h_G,
            num_clusters=num_clusters,
            model_name="EEE",
            random_state=seeds,
            blas_threads=blas_threads,
        )
        cluster_centers = torch.tensor(
            cluster_centers_np,
            dtype=torch.float32,
            device=encoder_device
        )

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=seeds)
        kmeans.fit(h_G)
        y_pred = kmeans.labels_

        cluster_centers_np = np.array([
            h_G[y_pred == cluster_id].mean(axis=0)
            for cluster_id in np.unique(y_pred)
        ])
        cluster_centers = torch.tensor(
            cluster_centers_np,
            dtype=torch.float32,
            device=encoder_device
        )

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return cluster_centers, y_pred