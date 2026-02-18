import logging

import numpy as np
import pandas as pd
import scanpy.pp as pp
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_ann.kneighbors.annoy import AnnoyTransformer
from scipy.sparse import csr_matrix
import marsilea as ma
import marsilea.plotter as mp
from tqdm import tqdm
from matplotlib.colors import to_rgb
import plotly.express as px

logger = logging.getLogger(__name__)


class UMAPHelper:
    def __init__(self, data=None, normalization="seurat"):
        # Use provided data or raise error
        if data is None:
            raise ValueError("No data provided")
        else:
            # Use provided DataFrame
            mask = ~data.isna().any(axis=1)
            clean_df = data.loc[mask].reset_index(drop=True)
            self.df = clean_df

        # Extract Cell IDs for coloring
        if "Cell ID" in self.df.columns:
            self.labels = self.df["Cell ID"].values
        else:
            self.labels = np.arange(1, len(self.df) + 1)
        # Extract feature columns (exclude Cell ID, Global X, Global Y, and
        # columns with "N/A")
        exclude_cols = ["Cell ID", "Global X", "Global Y"]
        self.feature_cols = [
            col
            for col in self.df.columns
            if col not in exclude_cols and not col.startswith("N/A")
        ]

        # Preprocess data
        self.data = self.df[self.feature_cols].values.astype(np.float64)

        # Create AnnData object
        self.data[self.data < 1] = 0
        self.adata = sc.AnnData(X=self.data)
        self.adata.obs["cell_id"] = self.labels
        self.adata.var_names = self.feature_cols
        total = len(self.data)
        sc.pp.filter_cells(self.adata, min_genes=1)
        self.adata.layers["counts"] = np.array(self.adata.X, dtype=np.int32)
        logger.info(f"Filtered {total-len(self.adata.X)} cells")

        if normalization == "pseudo-seurat":  # mentioned in tutorial
            sc.pp.normalize_total(self.adata)
            sc.pp.log1p(self.adata)
        elif normalization == "scale":  # used by video demonstrating seurat codex
            sc.pp.scale(self.adata)
        elif normalization == "seurat":
            sc.pp.normalize_total(self.adata, target_sum=10_000)
            sc.pp.log1p(self.adata)
            try:
                sc.pp.highly_variable_genes(
                    self.adata, flavor="seurat_v3", layer="counts", inplace=True)
            except Exception as e:
                logger.warning("HVG failed with default span, retrying with span=1.0: %s", e)
                sc.pp.highly_variable_genes(
                    self.adata,
                    flavor="seurat_v3",
                    layer="counts",
                    inplace=True,
                    span=1.0,
                )
            sc.pp.scale(self.adata, max_value=10)
        sc.pp.pca(self.adata)
        logger.info(
            f"Data loaded: {self.data.shape[0]} cells, {self.data.shape[1]} features")

    def getNumCells(self):
        return self.adata.n_obs

    def getNumFeatures(self):
        return self.adata.n_vars

    def getMaxPCs(self):
        return len(self.adata.varm["PCs"])

    def run_umap(
        self,
        n_components=20,
        pca_method="manual",
        variance_threshold=0.85,
        n_neighbors=15,
        min_dist=0.1,
        n_epochs=200,
        random_state=42,
        knn=False,
        kmeans=False,
        resolutions=[1.0],
    ):
        # Use pre-computed PCA, but subset to n_components
        self.resolutions = resolutions

        # --- Compute neighbors ---
        # if knn:
        sc.pp.neighbors(
            self.adata,
            transformer=AnnoyTransformer(n_trees=50, n_neighbors=n_neighbors),
            n_pcs=n_components,
        )
        # else:
        # sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_components, metric='cosine')
        # --- Leiden clustering for each resolution ---
        if len(resolutions) == 1:
            sc.tl.louvain(self.adata)
            sc.tl.paga(self.adata, groups="louvain")
            sc.tl.umap(self.adata, init_pos="paga")
        else:
            for res in resolutions:
                sc.tl.louvain(
                    self.adata,
                    key_added=f"leiden_res_{res:4.2f}",
                    resolution=res,
                    flavor="vtraag",
                    random_state=0,
                    directed=False,
                )

        # --- UMAP computation ---
        sc.tl.umap(
            self.adata,
            min_dist=min_dist,
            random_state=random_state,
            maxiter=n_epochs,
        )

        return self.adata

    def rank_genes(self, adata, top_n=10, figsize=(10, 7)):
        """
        Plot mean vs variance for all genes and label the top N highly variable genes.

        Parameters:
        -----------
        adata : AnnData object
            Annotated data object containing gene expression data
        top_n : int, default=10
            Number of top highly variable genes to label
        figsize : tuple, default=(10, 7)
            Figure size (width, height)

        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        """
        # Get top N highly variable genes by rank
        top_hvg = adata.var.nsmallest(top_n, "highly_variable_rank")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot all genes (unlabeled)
        ax.scatter(
            adata.var.means,
            adata.var.variances_norm,
            s=30,
            alpha=1.0,
            c="gray",
            label="All genes",
        )

        for gene_name, row in top_hvg.iterrows():
            ax.annotate(
                gene_name,
                xy=(row["means"], row["variances_norm"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        ax.set_xlabel("Mean expression", fontsize=12)
        ax.set_xscale("log")
        ax.set_ylabel("Normalized variance", fontsize=12)
        ax.set_title(
            f"Mean vs Variance (Top {top_n} Highly Variable Genes Labeled)",
            fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Print the top N genes
        logger.info(f"\nTop {top_n} Highly Variable Genes:")
        logger.info(top_hvg[["means", "variances_norm",
              "highly_variable_rank"]].to_string())
        return fig, ax

    def plot_umap(
        self,
        color="leiden",
        size=20,
        figsize=(10, 8),
        ax=None,
        show_centers=True,
        show=True,
        **kwargs,
    ):
        """Plot UMAP results with optional cluster center labels"""
        colors = [f"{color}_res_{i:.2f}" for i in self.resolutions]
        res = sc.pl.umap(
            self.adata,
            color=colors,
            show=show,
            legend_loc="on data",
        )
        if not show:
            return res
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, fig=None, key="leiden"):
        exp = self.adata.X
        m = ma.Heatmap(
            exp,
            cmap="viridis",
            height=20,
            width=10,
            vmin=0,
            vmax=3)
        m.group_rows(self.adata.obs[key])

        # Get cluster names and colors from adata
        cluster_names = list(self.adata.obs[key])

        # Assuming you have cluster colors stored in adata.uns
        if f"{key}_colors" in self.adata.uns:
            cluster_colors = self.adata.uns[f"{key}_colors"]
            unique_clusters = self.adata.obs[key].cat.categories
            cmapper = dict(zip(unique_clusters, cluster_colors, strict=False))
        else:
            raise Exception("run umap first")

        # Add cluster colors on the left
        colors = mp.Colors(cluster_names, palette=cmapper)
        m.add_left(colors, pad=0.1)

        # Add marker labels on top
        label_markers = mp.Labels(self.adata.var_names)
        m.add_top(label_markers, pad=0.1)

        m.add_legends()
        m.add_title("Expression Profile")
        m.render(fig)

    def show_colored_segmentation(
        self, seg, key="leiden", ignore=[], use_plotly=True, figsize=(12, 12)
    ):

        cluster_key = key

        # Verify the cluster key exists
        if cluster_key not in self.adata.obs.columns:
            raise ValueError(
                f"Cluster key '{cluster_key}' not found in adata.obs. "
                f"Available keys: {list(self.adata.obs.columns)}"
            )

        # Get leiden colors
        if f"{cluster_key}_colors" in self.adata.uns:
            leiden_colors = self.adata.uns[f"{cluster_key}_colors"]
            unique_leiden = self.adata.obs[cluster_key].cat.categories.astype(
                str)
        else:
            unique_leiden = np.sort(
                self.adata.obs[cluster_key].unique()).astype(str)
            n_clusters = len(unique_leiden)
            leiden_colors = sc.pl.palettes.default_102[:n_clusters]

        # Create leiden to RGB mapping
        leiden_to_color = {
            leiden: to_rgb(color)
            for leiden, color in zip(unique_leiden, leiden_colors, strict=False)
        }
        for i in ignore:
            leiden_to_color[i] = (0, 0, 0)

        # Find the maximum cell_id in BOTH segmentation and dataframe
        max_cell_id_df = self.adata.obs["cell_id"].max()
        max_cell_id_seg = seg.max()
        max_cell_id = max(max_cell_id_df, max_cell_id_seg)

        # Create lookup table: cell_id -> RGB color
        # Initialize with black (0, 0, 0) for cells not in dataframe
        color_lut = np.zeros((max_cell_id + 1, 3), dtype=np.float32)

        # Only color cells that exist in the dataframe (skip cell_id 0)
        valid_cell_ids = set(self.adata.obs["cell_id"])
        for cell_id, leiden in tqdm(
            zip(self.adata.obs["cell_id"], self.adata.obs[cluster_key]),
            desc="Mapping cell colors",
            total=len(self.adata.obs),
        ):
            if cell_id > 0 and cell_id <= max_cell_id:
                color_lut[cell_id] = leiden_to_color[str(leiden)]

        # Check for out-of-bounds cells in segmentation (excluding background)
        seg_cell_ids = set(np.unique(seg))
        seg_cell_ids.discard(0)  # Remove background
        valid_cell_ids.discard(0)  # Also ensure 0 is not in valid cells
        missing_cells = seg_cell_ids - valid_cell_ids

        if missing_cells:
            logger.debug(
                f"Warning: {len(missing_cells)} cell IDs in segmentation are not in dataframe"
            )
            logger.debug(
                f"These cells will be colored black: {sorted(list(missing_cells))[:10]}..."
                if len(missing_cells) > 10
                else f"These cells will be colored black: {sorted(missing_cells)}"
            )

        # Vectorized lookup
        colored_image = color_lut[seg]

        # Display the result
        if use_plotly:
            fig = px.imshow(colored_image)
            fig.update_layout(
                width=800,
                height=800,
                title=f"Segmentation colored by {cluster_key}")
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(colored_image)
            ax.set_title(f"Segmentation colored by {cluster_key}")
            ax.axis("off")
            plt.tight_layout()
            plt.show()
