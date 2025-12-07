import logging
import traceback

import matplotlib
import scanpy as sc
from PyQt6.QtCore import QThread, pyqtSignal
from ui.analysis.graphing.UMAPDataModel import DataModel

matplotlib.use("QtAgg")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIST_UMAP")


class AnalysisWorker(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        model: DataModel,
        n_neighbors,
        min_dist,
        n_components,
        random_state,
        resolution,
    ):
        super().__init__()
        self.model = model
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_state = random_state
        self.resolution = resolution

    def run(self):
        try:
            logger.info("Worker: Starting UMAP pipeline calculation...")
            adata, key = self.model.run_umap_pipeline(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=self.n_components,
                random_state=self.random_state,
                resolution=self.resolution,
            )
            logger.info("Worker: Pipeline calculation successful.")
            self.finished.emit(adata, key)

        except Exception as e:
            logger.exception("Worker failed during UMAP pipeline.")
            tb = traceback.format_exc()
            self.error.emit(f"Worker Error: {str(e)}\n\n{tb}")


class ClusteringWorker(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, model: DataModel, resolution):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def run(self):
        try:
            if self.model.adata is None:
                raise ValueError("No data loaded in model.")

            logger.info(
                f"Worker: Starting Re-clustering (Resolution={self.resolution})..."
            )

            key = "leiden"
            sc.tl.leiden(
                self.model.adata,
                resolution=self.resolution,
                key_added=key,
                flavor="igraph",
                n_iterations=2,
                random_state=42,
            )

            if f"{key}_colors" in self.model.adata.uns:
                del self.model.adata.uns[f"{key}_colors"]

            logger.info("Worker: Re-clustering successful.")
            self.finished.emit(self.model.adata, key)

        except Exception as e:
            logger.exception("Worker failed during clustering.")
            tb = traceback.format_exc()
            self.error.emit(f"Clustering Error: {str(e)}\n\n{tb}")
