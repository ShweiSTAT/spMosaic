"""Stage 3 spatial domain refinement after stage 2 for spMosaic."""

from __future__ import annotations


from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import anndata as ad


def create_adata_from_embed_meta(spot_embd: pd.DataFrame, spot_meta: pd.DataFrame) -> ad.AnnData:
    """Create an AnnData object from spot embeddings and spot metadata.

    This function constructs an :class:`anndata.AnnData` object using a spot
    embedding matrix and a corresponding metadata table. The two inputs are
    aligned by barcode, filtered to their shared barcodes, and used to build
    an AnnData object where:

    - ``X`` stores the embedding matrix
    - ``obs`` stores the aligned spot metadata
    - ``obsm["spatial"]`` stores spatial coordinates
    - ``obsm["X_embed"]`` stores a copy of the embedding matrix

    Parameters
    ----------
    spot_embd : pd.DataFrame
        Spot embedding matrix with barcodes as the index and embedding
        dimensions as columns.
    spot_meta : pd.DataFrame
        Spot metadata table. This must either contain a ``"barcode"`` column
        or use barcodes as the index. Expected metadata columns may include
        ``sample_name``, ``row``, ``col``, ``sum_umi``, ``domain_guess``,
        ``log_sum_umi``, ``DEC_initial_cluster``, and ``spatial_cluster``.

    Returns
    -------
    ad.AnnData
        AnnData object containing the embedding matrix, aligned metadata,
        and spatial coordinates.

    Raises
    ------
    ValueError
        If there are no overlapping barcodes between `spot_embd` and
        `spot_meta`, or if the spatial coordinate columns ``row`` and ``col``
        contain missing or non-numeric values.

    Notes
    -----
    The embedding matrix is stored both in ``adata.X`` and in
    ``adata.obsm["X_embed"]``.

    Spatial coordinates are stored in ``adata.obsm["spatial"]`` using the
    column order ``["col", "row"]``, following the convention that ``col``
    represents the x-coordinate and ``row`` represents the y-coordinate.

    Several metadata columns are optionally converted to categorical dtype if
    they are present in the metadata table.
    """
    # 1) Make sure barcodes are the index on both
    if 'barcode' in spot_meta.columns:
        spot_meta = spot_meta.drop_duplicates('barcode').set_index('barcode')
    else:
        spot_meta = spot_meta[~spot_meta.index.duplicated(keep='first')]

    spot_embd = spot_embd[~spot_embd.index.duplicated(keep='first')]

    # 2) Align by the intersection of barcodes (and keep the same order)
    common_barcodes = spot_embd.index.intersection(spot_meta.index)
    if len(common_barcodes) == 0:
        raise ValueError("No overlapping barcodes between spot_embd and spot_meta.")
    spot_embd = spot_embd.loc[common_barcodes].copy()
    spot_meta = spot_meta.loc[common_barcodes].copy()

    # 3) Ensure coordinates are numeric
    for c in ['row', 'col']:
        if c in spot_meta.columns:
            spot_meta[c] = pd.to_numeric(spot_meta[c], errors='coerce')
    if spot_meta[['row', 'col']].isnull().any().any():
        raise ValueError("Found non-numeric or missing values in 'row'/'col'.")

    # 4) Build AnnData
    var_names = [f"emb_{i+1}" for i in range(spot_embd.shape[1])]
    adata = ad.AnnData(
        X=spot_embd.to_numpy(dtype=float),
        obs=spot_meta,
        var=pd.DataFrame(index=var_names)
    )

    # 5) Put the spatial coordinates into obsm["spatial"]
    adata.obsm["spatial"] = spot_meta[['col', 'row']].to_numpy(dtype=float)

    # 6) (Optional) also keep the embedding in obsm
    adata.obsm["X_embed"] = spot_embd.to_numpy(dtype=float)

    # 7) (Optional) cast some obs columns to categorical
    for cat_col in ["sample_name", "domain_guess", "DEC_initial_cluster", "spatial_cluster"]:
        if cat_col in adata.obs:
            adata.obs[cat_col] = adata.obs[cat_col].astype('category')

    return adata


def batch_refine_label(
    adata,
    n_neighbors: int = 30,
    key: str = "label",
    batch_key: str = "batchID",
    suffix: str | None = None,
    tie_break: str = "keep",
    metric: str = "euclidean",
    max_iter: int = 10,
    min_change_frac: float = 1e-2,
    verbose: bool = True,
    random_state: int | None = 0
):
    """Iteratively refine spot labels within each batch using spatial majority vote.

    This function smooths discrete labels by replacing each spot's label with
    the majority label among its spatial neighbors, while restricting the
    refinement to occur separately within each batch. The spatial neighbor
    graph is built once per batch from ``adata.obsm["spatial"]`` and remains
    fixed across iterations.

    At each iteration, labels are updated by majority vote among neighboring
    spots in the same batch. The procedure stops when either:

    - no labels change, or
    - the fraction of changed labels is less than or equal to
      ``min_change_frac``, or
    - the number of iterations reaches ``max_iter``.

    The refined labels are written to a new column in ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial coordinates in
        ``adata.obsm["spatial"]`` and label metadata in ``adata.obs``.
    n_neighbors : int, default=30
        Number of spatial neighbors used for majority voting within each
        batch. If a batch contains fewer spots than this value, the effective
        number of neighbors is reduced automatically.
    key : str, default="label"
        Name of the column in ``adata.obs`` containing the labels to refine.
    batch_key : str, default="batchID"
        Name of the column in ``adata.obs`` defining the batch membership used
        to separate refinement across batches.
    suffix : str or None, default=None
        Optional suffix appended to the output column name.
        The output column is named ``f"{key}_refined"`` if `suffix` is None,
        otherwise ``f"{key}_refined_{suffix}"``.
    tie_break : str, default="keep"
        Strategy used when multiple labels are tied for the majority vote.

        - ``"keep"``: keep the current label
        - ``"random"``: randomly choose one of the tied labels
    metric : str, default="euclidean"
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`
        when constructing the spatial neighbor graph.
    max_iter : int, default=10
        Maximum number of refinement iterations.
    min_change_frac : float, default=1e-2
        Convergence threshold based on the fraction of labels changed in an
        iteration.
    verbose : bool, default=True
        Whether to print iteration progress and convergence messages.
    random_state : int or None, default=0
        Random seed used when ``tie_break="random"``. If None, a fresh random
        generator is used.

    Returns
    -------
    dict
        Dictionary summarizing the refinement process with keys:

        - ``"column"``: name of the output column written to ``adata.obs``
        - ``"iterations"``: number of iterations performed
        - ``"history"``: list of dictionaries recording the number and
          fraction of changed labels per iteration

    Raises
    ------
    ValueError
        If ``adata.obsm["spatial"]`` is missing, or if the required columns
        `key` and `batch_key` are not found in ``adata.obs``.

    Notes
    -----
    This function operates on encoded integer label codes internally for
    efficiency, then decodes the refined labels back to their original values.

    If the original labels are categorical, the output column is also stored
    as a categorical series with the same category set.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    if key not in adata.obs or batch_key not in adata.obs:
        raise ValueError(f"obs['{key}'] or obs['{batch_key}'] not found.")

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        np.random.seed(random_state)
    else:
        rng = np.random.default_rng()

    coords = adata.obsm["spatial"]
    labels = adata.obs[key]
    obs_index = adata.obs.index

    # ---- Encode labels to integer codes (and remember mapping) ----
    if pd.api.types.is_categorical_dtype(labels):
        cat = labels.cat
        code_to_label = np.array(cat.categories)
        global_codes = cat.codes.to_numpy()  # -1 means NaN
        categorical_out = True
    else:
        uniq = pd.Index(labels.astype("object").unique())
        label_to_code = {lab: i for i, lab in enumerate(uniq)}
        code_to_label = uniq.to_numpy()
        global_codes = labels.astype("object").map(label_to_code).to_numpy()
        categorical_out = False

    # ---- Build per-batch neighbor indices once (static graph) ----
    batch_groups = adata.obs.groupby(batch_key, sort=False).groups
    per_batch = {}
    for bk, idx_names in batch_groups.items():
        idx_names = pd.Index(idx_names)
        idx_pos = obs_index.get_indexer(idx_names)
        idx_pos = idx_pos[idx_pos >= 0]
        if idx_pos.size <= 1:
            per_batch[bk] = {"idx_pos": idx_pos, "neigh_ind": None}
            continue

        X = coords[idx_pos]
        k = int(min(max(n_neighbors, 1), max(1, X.shape[0] - 1)))
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nn.fit(X)
        ind = nn.kneighbors(return_distance=False)[:, 1:]
        per_batch[bk] = {"idx_pos": idx_pos, "neigh_ind": ind}

    # ---- Iterate until convergence ----
    refined = np.array(global_codes, copy=True)
    n_total = refined.size
    change_history = []

    for it in range(1, max_iter + 1):
        changed = 0
        new_refined = np.array(refined, copy=True)

        for bk, bundle in per_batch.items():
            idx_pos = bundle["idx_pos"]
            ind = bundle["neigh_ind"]
            if idx_pos.size <= 1 or ind is None:
                continue

            y_codes = refined[idx_pos]
            neigh_codes_all = y_codes[ind]

            for i_local in range(neigh_codes_all.shape[0]):
                neigh_codes = neigh_codes_all[i_local]
                neigh_codes = neigh_codes[neigh_codes >= 0]
                if neigh_codes.size == 0:
                    continue

                counts = np.bincount(neigh_codes)
                winners = np.flatnonzero(counts == counts.max())

                if winners.size == 1:
                    proposed = winners[0]
                else:
                    proposed = (
                        y_codes[i_local]
                        if tie_break == "keep"
                        else rng.choice(winners)
                    )

                if proposed != refined[idx_pos[i_local]]:
                    new_refined[idx_pos[i_local]] = proposed
                    changed += 1

        refined = new_refined
        frac = changed / n_total
        change_history.append({"iter": it, "changed": changed, "frac_changed": frac})

        if verbose:
            print(f"[Refine] Iter {it:02d}: changed={changed} ({frac:.4f})")

        if changed == 0 or frac <= min_change_frac:
            if verbose:
                print("[Refine] Converged.")
            break

    # ---- Decode back to labels and write to obs ----
    refined_labels = pd.Series(code_to_label[refined], index=obs_index)
    if categorical_out:
        refined_labels = refined_labels.astype(
            pd.CategoricalDtype(categories=code_to_label, ordered=False)
        )

    suffix_add = "" if not suffix else f"_{suffix}"
    out_col = f"{key}_refined{suffix_add}"
    adata.obs[out_col] = refined_labels

    return {
        "column": out_col,
        "iterations": len(change_history),
        "history": change_history,
    }