import logging

import numpy as np
import scanpy as sc

# Setup logger for this module
logger = logging.getLogger(__name__)


class DataModel:
    def __init__(self, data=None, normalization="lognorm"):
        if data is None:
            raise ValueError("No data provided")

        # Store the immutable raw data
        self.raw_df = data
        self.normalization_method = normalization

        # Define available features from the raw data immediately
        exclude_cols = ["Cell ID", "Global X", "Global Y"]
        self.all_features = [
            col
            for col in self.raw_df.columns
            if col not in exclude_cols and not col.startswith("N/A")
        ]

        # Default: Use all features
        self.selected_features = self.all_features.copy()

        self.adata = None
        self.hvg_rankings = None

        # Initial Build
        self.reprocess_data(self.normalization_method, self.selected_features)

    def get_all_features(self):
        """Returns list of all available columns from raw dataframe"""
        return self.all_features

    def reprocess_data(self, normalization, features):
        """
        Rebuilds the AnnData object from scratch using specific features and normalization.
        This is a 'destructive' update (resets PCA/UMAP).
        """
        logger.info(
            f"Reprocessing data: Norm={normalization}, Features={len(features)}"
        )

        self.normalization_method = normalization
        self.selected_features = features

        # 1. Filter Raw DF by Selected Features
        # Always include metadata like Cell ID
        cols_to_use = (
            ["Cell ID"] + features if "Cell ID" in self.raw_df.columns else features
        )
        subset_df = self.raw_df[cols_to_use].copy()

        # 2. Remove NaNs
        mask = ~subset_df.isna().any(axis=1)
        clean_df = subset_df.loc[mask].reset_index(drop=True)

        # 3. Create AnnData
        if "Cell ID" in clean_df.columns:
            labels = clean_df["Cell ID"].values
            data_values = clean_df[features].values.astype(np.float64)
        else:
            labels = np.arange(1, len(clean_df) + 1)
            data_values = clean_df.values.astype(np.float64)
        if normalization != "none":
            data_values[data_values < 1] = 0

        self.adata = sc.AnnData(X=data_values)
        self.adata.obs["cell_id"] = labels
        self.adata.var_names = features

        # 4. Standard Filtering
        sc.pp.filter_cells(self.adata, min_genes=1)
        self.adata.layers["counts"] = self.adata.X.copy()

        # 5. Apply Selected Normalization
        self._apply_normalization()

        # 6. Recompute PCA
        self._compute_hvg_and_pca()

        logger.info(f"Data reprocessed. Shape: {self.adata.shape}")

    def get_normalization_methods(self):
        """Returns the list of available normalization methods"""
        return ["RC", "lognorm", "znorm", "none"]

    def _apply_normalization(self):
        """Applies the specific normalization logic defined in original code"""
        assert isinstance(self.adata, sc.AnnData), "adata is not an AnnData instance"
        if self.normalization_method == "RC":
            sc.pp.normalize_total(self.adata)
            sc.pp.scale(self.adata, max_value=10)
        elif self.normalization_method == "lognorm":
            logger.info("Applying log Normalization")
            sc.pp.normalize_total(self.adata, target_sum=10_000)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, max_value=10)
        elif self.normalization_method == "none":
            logger.info("No normalization applied.")
        elif self.normalization_method == "znorm":
            X = self.adata.X.astype(float)
            means = X.mean(axis=0)
            stds = X.std(axis=0)
            self.adata.X = (X - means) / stds
            # sc.pp.scale(self.adata)
            logger.info("Z-normalization applied.")

    def _compute_hvg_and_pca(self):
        """Computes Highly Variable Genes and initial PCA"""
        # HVG Logic
        assert isinstance(self.adata, sc.AnnData), "adata is not an AnnData instance"
        sc.pp.pca(self.adata)
        try:
            sc.pp.highly_variable_genes(
                self.adata, flavor="seurat_v3", layer="counts", inplace=True
            )
        except Exception as e:
            logger.warning(
                f"Seurat v3 HVG failed, falling back to span=1.0. Error: {e}"
            )
            sc.pp.highly_variable_genes(
                self.adata, flavor="seurat_v3", layer="counts", inplace=True, span=1.0
            )

        # Save rankings for UI visualization
        if "highly_variable_rank" in self.adata.var.columns:
            self.hvg_rankings = self.adata.var["highly_variable_rank"].copy()

        # PCA

    # --- Public Accessors for UI ---
    def get_num_cells(self):
        assert self.adata is not None, "adata is not initialized"
        return self.adata.n_obs

    def get_num_features(self):
        assert self.adata is not None, "adata is not initialized"
        return self.adata.n_vars

    def get_max_pcs(self):
        """Returns the maximum number of PCs available for UMAP selection"""
        assert self.adata is not None, "adata is not initialized"
        return len(self.adata.varm["PCs"]) if "PCs" in self.adata.varm else 0

    def get_pca_variance_ratio(self):
        """Returns the variance ratio array for the 'Variance Threshold' UI logic"""
        assert self.adata is not None, "adata is not initialized"
        return self.adata.uns["pca"]["variance_ratio"]

    # --- Heavy Calculation Methods (To be called by Worker Thread) ---

    def run_clustering_only(self, resolution):
        """Runs only Leiden clustering (fast update for slider)"""
        logger.info(f"Re-running clustering at resolution {resolution}")
        assert isinstance(self.adata, sc.AnnData), "adata is not initialized"
        key = "leiden"
        sc.tl.leiden(
            self.adata,
            key_added=key,
            resolution=resolution,
            # flavor="vtraag",
            flavor="igraph",
            n_iterations=2,
            random_state=0,
            directed=False,
        )
        return self.adata, key

    def run_umap_pipeline(
        self, n_neighbors, min_dist, n_components, random_state, resolution
    ):
        """
        Runs Neighbors -> UMAP -> Leiden.
        Replicates run_umap logic but is designed to be atomic for the thread.
        """
        logger.info(
            f"Running UMAP Pipeline: neighbors={n_neighbors}, dist={min_dist}, pcs={n_components}"
        )
        assert isinstance(self.adata, sc.AnnData), "adata is not initialized"

        # 1. Compute Neighbors (using AnnoyTransformer as originally requested)
        sc.pp.neighbors(
            self.adata,
            # transformer=AnnoyTransformer(n_trees=50, n_neighbors=n_neighbors),  # type: ignore
            random_state=random_state,
            n_neighbors=n_neighbors,
            n_pcs=n_components,
        )

        # 2. Run UMAP
        sc.tl.umap(
            self.adata,
            min_dist=min_dist,
            random_state=random_state,
            maxiter=None,  # removed restriction as per original 'n_epochs=None' comment
        )

        # 3. Initial Clustering
        adata, key = self.run_clustering_only(resolution)

        return adata, key
