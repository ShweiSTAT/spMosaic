"""Stage 2 spatial domain detection functions for spMosaic."""

from __future__ import annotations

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import scanpy as sc

import copy
import os

from .model import DualEncoderAutoencoder, loss_function, DEC
from .clustering import initialize_clusters
from .cluster_refine import create_adata_from_embed_meta, batch_refine_label
from .utils import set_seed


if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_autoencoder(
    X_dataset,
    batch_labels,
    input_dim,
    embedding_dim,
    batch_num,
    epochs=100,
    learning_rate=0.001,
    fuse_weight=1,
    patience=5,
    min_delta=0.05,
    test_split_rate=0.2,
):
    """Pretrain the dual-encoder autoencoder.

    This function trains the :class:`DualEncoderAutoencoder` using the full
    input expression matrix and batch labels. The training objective combines:

    - reconstruction loss for the input expression
    - batch-classification loss for the batch embedding branch

    The model is trained with Adam optimization and a learning-rate scheduler
    based on validation loss. Early stopping is applied when the validation
    loss fails to improve by at least ``min_delta`` for ``patience`` checks.

    Parameters
    ----------
    X_dataset : np.ndarray or torch.Tensor
        Input expression matrix of shape ``(n_samples, input_dim)``.
    batch_labels : np.ndarray or torch.Tensor
        Batch labels of shape ``(n_samples,)``.
    input_dim : int
        Number of input features.
    embedding_dim : int
        Dimension of each latent embedding branch.
    batch_num : int
        Number of batch classes.
    epochs : int, default=100
        Maximum number of training epochs.
    learning_rate : float, default=0.001
        Initial learning rate for Adam.
    fuse_weight : float, default=1
        Weight applied to the batch-classification loss.
    patience : int, default=5
        Number of consecutive checks without sufficient validation improvement
        before early stopping.
    min_delta : float, default=0.05
        Minimum relative improvement in validation loss required to reset the
        early stopping counter.
    test_split_rate : float, default=0.2
        Fraction of data used as the validation set.

    Returns
    -------
    model : DualEncoderAutoencoder
        The trained autoencoder with the best validation-state weights loaded.

    Notes
    -----
    The function automatically selects the best available device in the order:
    CUDA, MPS, then CPU.

    The returned model is restored to the best validation-loss state observed
    during training.
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset,
        batch_labels,
        test_size=test_split_rate,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create model
    model = DualEncoderAutoencoder(input_dim, embedding_dim, batch_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Lists to store losses
    train_recon_losses, train_class_losses, train_total_losses = [], [], []
    test_recon_losses, test_class_losses, test_total_losses = [], [], []

    # Early stopping variables
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    prev_lr = learning_rate

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_reconstructed, batch_pred, h_G, h_B = model(X_train)
        recon_loss, class_loss, total_loss = loss_function(
            X_train, x_reconstructed, y_train, batch_pred, fuse_weight
        )
        total_loss.backward()
        optimizer.step()

        train_recon_losses.append(recon_loss.item())
        train_class_losses.append(class_loss.item())
        train_total_losses.append(total_loss.item())

        model.eval()
        with torch.no_grad():
            x_reconstructed_test, batch_pred_test, _, _ = model(X_test)
            recon_loss_test, class_loss_test, total_loss_test = loss_function(
                X_test, x_reconstructed_test, y_test, batch_pred_test, fuse_weight
            )

        test_recon_losses.append(recon_loss_test.item())
        test_class_losses.append(class_loss_test.item())
        test_total_losses.append(total_loss_test.item())

        scheduler.step(total_loss_test)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr != prev_lr:
            print(
                f"Learning rate changed at epoch {epoch + 1}: "
                f"{prev_lr:.6f} → {current_lr:.6f}"
            )
            prev_lr = current_lr

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}, Training Loss - Recon: {recon_loss:.4f} | "
                f"Class: {class_loss:.4f} | Total: {total_loss:.4f}"
            )
            print(
                f"Epoch {epoch + 1}, Testing Loss - Recon: {recon_loss_test:.4f} | "
                f"Class: {class_loss_test:.4f} | Total: {total_loss_test:.4f}"
            )

        if total_loss_test < best_loss * (1 - min_delta):
            best_loss = total_loss_test
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch - patience + 1}. "
                f"Best validation loss: {best_loss:.4f}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_DEC(
    data,
    autoencoder,
    num_clusters,
    gamma=0.1,
    method="gmm",
    epochs=100,
    learning_rate=0.01,
    test_size=0.2,
    stop_rate=0.01,
    seeds=2020,
    batch_size=256,
    blas_threads: int | None = 1,
):
    """Train the DEC model for spatial domain refinement.

    This function performs DEC-based clustering refinement using a pretrained
    dual-encoder autoencoder. The biological encoder is used to obtain an
    initial embedding, cluster centers are initialized by a separate
    clustering method, and the DEC model is then optimized using a combined
    objective consisting of:

    - reconstruction loss for the input expression
    - KL divergence between the soft cluster assignment distribution and the
      DEC target distribution

    Training is carried out with mini-batches on the training split, while
    the full dataset is used to monitor cluster assignment changes across
    epochs. Training stops early when the cluster assignment change rate falls
    below ``stop_rate``.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input expression matrix of shape ``(n_samples, input_dim)``.
    autoencoder : DualEncoderAutoencoder
        Pretrained dual-encoder autoencoder used to initialize the DEC model.
    num_clusters : int
        Number of clusters.
    gamma : float, default=0.1
        Weight applied to the KL divergence term in the DEC loss.
    method : str, default="gmm"
        Method used to initialize clusters. Must be one of the methods
        supported by :func:`initialize_clusters`, such as ``"gmm"``,
        ``"mclust"``, or ``"kmeans"``.
    epochs : int, default=100
        Maximum number of DEC training epochs.
    learning_rate : float, default=0.01
        Initial learning rate for Adam optimization.
    test_size : float, default=0.2
        Fraction of data reserved for the test split.
    stop_rate : float, default=0.01
        Early stopping threshold for the cluster assignment change rate.
        Training stops when the change rate falls below this value.
    seeds : int, default=2020
        Random seed used for cluster initialization.
    batch_size : int, default=256
        Mini-batch size used during DEC training.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries when cluster initialization uses the R-based ``mclust``
        subprocess. Setting this to 1 helps avoid hidden multithreading and
        oversubscription on HPC systems. Ignored for non-R initialization
        methods such as ``"gmm"`` and ``"kmeans"``. If None, existing
        environment settings are left unchanged.

    Returns
    -------
    dec_model : DEC
        Trained DEC model.
    y_pred_init : np.ndarray
        Initial cluster assignments obtained before DEC refinement.

    Raises
    ------
    ValueError
        If ``blas_threads`` is less than 1 when not None.

    Notes
    -----
    The function automatically selects the best available device in the order:
    CUDA, MPS, then CPU.

    The learning-rate scheduler is updated using the cluster assignment change
    rate rather than the validation loss.
    """
    if blas_threads is not None and blas_threads < 1:
        raise ValueError("`blas_threads` must be at least 1 or None.")

    # Split data into train/test
    train_data_np, test_data_np = train_test_split(
        data, test_size=test_size, random_state=42
    )
    train_tensor = torch.tensor(train_data_np, dtype=torch.float32)
    test_tensor = torch.tensor(test_data_np, dtype=torch.float32).to(device)
    full_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Mini-batch dataloader
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # Move model to device
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Initialize clusters
    cluster_centers, y_pred_init = initialize_clusters(
        autoencoder.encoder_G,
        full_tensor,
        num_clusters,
        method,
        seeds,
        blas_threads=blas_threads,
    )

    # Initialize DEC
    dec_model = DEC(
        autoencoder,
        embedding_dim=autoencoder.encoder_G[-1].out_features,
        num_clusters=num_clusters,
        cluster_centers=cluster_centers
    ).to(device)

    optimizer = torch.optim.Adam(dec_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    loss_fn_recon = torch.nn.functional.mse_loss
    loss_fn_kl = torch.nn.KLDivLoss(reduction='batchmean')

    # Logs
    train_losses, test_losses = [], []
    train_rec_losses, test_rec_losses = [], []
    train_kl_losses, test_kl_losses = [], []
    cluster_change_rates = []
    previous_all_preds = y_pred_init
    prev_lr = learning_rate

    for epoch in range(epochs):
        dec_model.train()
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()

            h_G_batch, q_batch, p_batch, recon_batch = dec_model(x_batch)
            loss_recon = loss_fn_recon(recon_batch, x_batch)
            loss_kl = loss_fn_kl(q_batch.log(), p_batch)
            total_loss = loss_recon + gamma * loss_kl

            total_loss.backward()
            optimizer.step()

            batch_size_actual = x_batch.size(0)
            total_samples += batch_size_actual
            epoch_recon_loss += loss_recon.item() * batch_size_actual
            epoch_kl_loss += loss_kl.item() * batch_size_actual
            epoch_total_loss += total_loss.item() * batch_size_actual

        # Normalize losses
        train_rec_losses.append(epoch_recon_loss / total_samples)
        train_kl_losses.append(epoch_kl_loss / total_samples)
        train_losses.append(epoch_total_loss / total_samples)

        # Evaluation
        dec_model.eval()
        with torch.no_grad():
            h_G_all, q_all, p_all, recon_all = dec_model(full_tensor)
            cluster_assignments_all = torch.argmax(q_all, dim=1).cpu().numpy()

            _, q_test, p_test, recon_test = dec_model(test_tensor)
            loss_recon_test = loss_fn_recon(recon_test, test_tensor)
            loss_kl_test = loss_fn_kl(q_test.log(), p_test)
            total_loss_test = loss_recon_test + gamma * loss_kl_test

            test_rec_losses.append(loss_recon_test.item())
            test_kl_losses.append(loss_kl_test.item())
            test_losses.append(total_loss_test.item())

            if previous_all_preds is not None:
                change_rate = np.mean(cluster_assignments_all != previous_all_preds)
            else:
                change_rate = 1.0

            cluster_change_rates.append(change_rate)
            previous_all_preds = cluster_assignments_all

            scheduler.step(change_rate)
            current_lr = scheduler.get_last_lr()[0]
            if current_lr != prev_lr:
                print(
                    f"Learning rate changed at epoch {epoch + 1}: "
                    f"{prev_lr:.6f} → {current_lr:.6f}"
                )
                prev_lr = current_lr

            if epoch % 5 == 0:
                print(f"Epoch {epoch}:")
                print(
                    f"  Train Total: {train_losses[-1]:.4f} | "
                    f"Recon: {train_rec_losses[-1]:.4f} | "
                    f"KL: {train_kl_losses[-1]:.4f}"
                )
                print(
                    f"  Test  Total: {total_loss_test:.4f} | "
                    f"Recon: {loss_recon_test:.4f} | "
                    f"KL: {loss_kl_test:.4f}"
                )
                print(f"  Cluster Change Rate: {change_rate:.5f}")

            if change_rate < stop_rate:
                print(
                    f"Early stopping at epoch {epoch} — "
                    f"change rate {change_rate:.5f} < {stop_rate}"
                )
                break

    return dec_model, y_pred_init


def domain_detection(
    h5ad_path,
    input_dir,
    prefix,
    n_clusters,
    clust_method="mclust",
    SVG_selection_method="common",
    lambda_b=1e0,
    alpha=1e-3,
    fix_seed=888,
    if_cluster_refine=True,
    clust_refine_k_num=30,
    pre_train_epochs=500,
    DEC_epochs=10,
    blas_threads: int | None = 1,
):
    """Run stage 2 spatial domain detection from GAM-smoothed gene expression.

    This function performs the second stage of the spMosaic pipeline. It reads
    the outputs produced by stage 1 gene smoothing, selects a set of genes for
    downstream analysis, trains a dual-encoder autoencoder, refines cluster
    assignments using DEC, optionally performs spatial KNN-based label
    refinement, and returns a final AnnData object containing the learned
    embeddings and spatial domain labels.

    The main steps are:

    1. read GAM-smoothed expression, spot metadata, and combined GAM test
       statistics
    2. select genes based on the requested SVG selection rule
    3. standardize expression within each sample
    4. pretrain the dual-encoder autoencoder
    5. initialize clusters and run DEC refinement
    6. optionally refine spatial labels using within-sample KNN smoothing
    7. save embedding and metadata outputs and return an updated AnnData object

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the original input ``.h5ad`` file. This file is reloaded at the
        end so that the final embeddings and predicted spatial labels can be
        written back into an AnnData object.
    input_dir : str or Path
        Directory containing the stage 1 GAM output folder
        ``{prefix}_gam_output``.
    prefix : str
        Prefix identifying the current analysis. This is used to locate GAM
        outputs and to name stage 2 output files.
    n_clusters : int
        Number of clusters used for DEC initialization and refinement.
    clust_method : str, default="mclust"
        Method used for DEC cluster initialization. Supported options depend on
        :func:`initialize_clusters`, such as ``"mclust"``, ``"gmm"``, and
        ``"kmeans"``.
    SVG_selection_method : str, default="common"
        Rule used to select genes from the GAM output. Supported options are:

        - ``"common"``: use genes marked as common SVGs
        - ``"union"``: use genes marked as union SVGs
        - ``"T2000"``, ``"T3000"``, ``"T4000"``, ``"T5000"``: use the top-ranked
          genes from the GAM combined statistics table
    lambda_b : float, default=1e0
        Weight applied to the batch-classification loss during autoencoder
        pretraining.
    alpha : float, default=1e-4
        Weight applied to the DEC KL-divergence loss during cluster refinement.
    fix_seed : int, default=888
        Random seed used for reproducibility in Python-based steps and in
        downstream clustering procedures.
    if_cluster_refine : bool, default=True
        Whether to apply an additional spatial KNN-based label refinement step
        after DEC clustering.
    clust_refine_k_num : int, default=30
        Number of spatial neighbors used in the optional KNN refinement step.
    pre_train_epochs : int, default=500
        Maximum number of epochs for autoencoder pretraining.
    DEC_epochs : int, default=10
        Maximum number of epochs for DEC refinement.
    blas_threads : int or None, default=1
        Number of threads allowed for low-level BLAS/OpenMP numerical
        libraries in R subprocess calls during stage 2. This currently affects
        cluster initialization when ``clust_method="mclust"``. Setting this to
        1 helps avoid hidden multithreading and unexpected slowdown on HPC
        systems. Ignored for non-R initialization methods such as ``"gmm"``
        and ``"kmeans"``. If None, existing environment settings are left
        unchanged.

    Returns
    -------
    anndata.AnnData
        AnnData object containing:

        - ``obsm["spmosaic_embd"]``: learned biological embedding
        - ``obsm["spatial"]``: spatial coordinates
        - ``obs``: updated metadata including initial and refined cluster labels

    Raises
    ------
    ValueError
        If ``blas_threads`` is less than 1 when not None.

    Notes
    -----
    This function expects that stage 1 has already produced the following files
    under ``{input_dir}/{prefix}_gam_output``:

    - ``{prefix}_smoothed_gene_exp.csv``
    - ``{prefix}_spots_metadata.csv``
    - ``{prefix}_GAM_CombinedStat.csv``

    The function writes stage 2 outputs to ``{input_dir}/{prefix}_dec_output``,
    including spot embeddings and spot metadata as CSV files.

    If spatial refinement is enabled, the refined labels are stored in the
    AnnData object under a column such as ``"spatial_cluster_refined"``.
    """
    if blas_threads is not None and blas_threads < 1:
        raise ValueError("`blas_threads` must be at least 1 or None.")

    print("Device being used is: " + str(device))

    ## loading data
    print("Step 2-1: Reading in GAM output for spatial domain detection...")
    # smoothed gene expression
    PredY_path = os.path.join(input_dir, f"{prefix}_gam_output", f"{prefix}_smoothed_gene_exp.csv")
    PredY = pd.read_csv(PredY_path, index_col=0)
    PredY = PredY.T

    # spot meta data
    metadata_path = os.path.join(input_dir, f"{prefix}_gam_output", f"{prefix}_spots_metadata.csv")
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata["barcode"] = metadata.index
    metadata["sample_name"] = metadata["sample_name"].astype(str)

    # combined test stats
    testSTAT_path = os.path.join(input_dir, f"{prefix}_gam_output", f"{prefix}_GAM_CombinedStat.csv")
    testSTAT = pd.read_csv(testSTAT_path, index_col=0)
    testSTAT["genes"] = testSTAT.index

    ## select a gene set to proceed
    common_svgs = testSTAT[testSTAT["if_common_SVGs"] == True]
    union_svgs = testSTAT[testSTAT["if_union_SVGs"] == True]

    print("Common_svgs number: " + str(len(common_svgs)))
    print("Union_svgs number: " + str(len(union_svgs)))
    print("Gene selection method used is: " + SVG_selection_method)

    if SVG_selection_method == "common":
        genes_to_select = common_svgs["genes"]
    elif SVG_selection_method == "union":
        genes_to_select = union_svgs["genes"]
    elif SVG_selection_method == "T2000":
        genes_to_select = testSTAT["genes"].iloc[:2000]
    elif SVG_selection_method == "T3000":
        genes_to_select = testSTAT["genes"].iloc[:3000]
    elif SVG_selection_method == "T4000":
        genes_to_select = testSTAT["genes"].iloc[:4000]
    elif SVG_selection_method == "T5000":
        genes_to_select = testSTAT["genes"].iloc[:5000]
    else:
        raise ValueError(f"Unknown SVG_selection_method: {SVG_selection_method}")

    ## Subset PredY_sample with selected genes
    PredY_subset = PredY.loc[:, PredY.columns.intersection(genes_to_select)].copy()

    ## data standardization
    expr_cols = PredY_subset.columns
    combined_df = PredY_subset.copy()
    combined_df["sample_name"] = metadata["sample_name"].values
    standardized_df = (
        combined_df.groupby("sample_name")[expr_cols]
        .apply(lambda x: (x - x.mean()) / x.std())
    )

    ## parameter settings
    standardized_df.index = standardized_df.index.get_level_values(1)
    standardized_df = standardized_df.loc[metadata["barcode"]]
    Gene_exp_data = standardized_df.values
    unique_batches, batch_labels_int = np.unique(
        combined_df["sample_name"].values,
        return_inverse=True
    )

    input_dim = Gene_exp_data.shape[1]
    embedding_dim = 32
    batch_num = len(unique_batches)

    set_seed(fix_seed)

    ## Autoencoder training
    print("Step 2-2: Pretraning dual-encoder autoencoder ...")
    model_preTrain = train_autoencoder(
        Gene_exp_data,
        batch_labels_int,
        input_dim,
        embedding_dim,
        batch_num,
        epochs=pre_train_epochs,
        learning_rate=1e-4,
        fuse_weight=lambda_b,
        patience=10,
        min_delta=1e-3,
        test_split_rate=0.2
    )

    pre_trained_autoencoder = copy.deepcopy(model_preTrain)

    print("Step 2-3: DEC refinement ...")
    dec_model, y_pred_init = train_DEC(
        Gene_exp_data,
        pre_trained_autoencoder,
        num_clusters=n_clusters,
        learning_rate=1e-4,
        gamma=alpha,
        method=clust_method,
        test_size=0.1,
        epochs=DEC_epochs,
        stop_rate=1e-2,
        seeds=fix_seed,
        batch_size=256 * 2 * 2,
        blas_threads=blas_threads,
    )

    #######################
    ## get cluster centers
    #######################

    ## extract pre-train results
    data_genes_tensor = torch.tensor(Gene_exp_data, dtype=torch.float32).to(device)
    dec_model.eval()
    with torch.no_grad():
        h_G_all, q_all, p_all, recon_all = dec_model(data_genes_tensor)

    latent_spot_debatch_numpy = h_G_all.detach().cpu().numpy()
    cluster_assignments_final = torch.argmax(q_all, dim=1).cpu().numpy()

    metadata["DEC_init_cluster"] = y_pred_init
    metadata["spatial_cluster"] = cluster_assignments_final

    spot_meta = metadata.copy()
    spot_embd = pd.DataFrame(latent_spot_debatch_numpy, index=metadata["barcode"])

    ## cluster refinement by KNN
    adata = create_adata_from_embed_meta(spot_embd, spot_meta)
    if if_cluster_refine:
        print("Step 2-4: cluster refinement by spatial KNN ...")
        batch_refine_label(
            adata,
            n_neighbors=clust_refine_k_num,
            key="spatial_cluster",
            batch_key="sample_name",
            random_state=fix_seed
        )

    ## output
    spot_embd = pd.DataFrame(h_G_all.cpu().numpy(), index=metadata["barcode"])
    spot_meta = adata.obs.copy()
    wanted_spot_meat_cols = [
        "sample_name", "row", "col", "sum_umi",
        "DEC_init_cluster", "spatial_cluster", "spatial_cluster_refined"
    ]
    selected_cols = [c for c in wanted_spot_meat_cols if c in spot_meta.columns]
    spot_meta = spot_meta.loc[:, selected_cols].copy()

    outs = {
        "Spot_embd": spot_embd,
        "Spot_meta": spot_meta,
    }

    DEC_out_path = os.path.join(input_dir, f"{prefix}_dec_output")
    os.makedirs(DEC_out_path, exist_ok=True)

    Spot_embd_path = os.path.join(DEC_out_path, f"{prefix}_spot_embd.csv")
    Spot_meta_path = os.path.join(DEC_out_path, f"{prefix}_spot_meta.csv")

    print("outting: " + Spot_embd_path)
    print("outting: " + Spot_meta_path)
    outs["Spot_embd"].to_csv(Spot_embd_path)
    outs["Spot_meta"].to_csv(Spot_meta_path)

    h5ad_out = sc.read_h5ad(h5ad_path)
    h5ad_out.obsm["spmosaic_embd"] = spot_embd.to_numpy(dtype=float)
    h5ad_out.obsm["spatial"] = spot_meta[["col", "row"]].to_numpy(dtype=float)
    h5ad_out.obs = spot_meta

    print("All done for stage 2!")
    return h5ad_out