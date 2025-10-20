import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt
from sklearn.datasets import make_blobs
import umap
import umap.plot

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler,MinMaxScaler
class UMAPVisualizer(QMainWindow):
    def __init__(self, data=None):
        super().__init__()

        self.setWindowTitle("UMAP Visualizer")
        self.setGeometry(100, 100, 800, 600)
        
        # Use provided data or generate synthetic data
        if data is None:
            raise ValueError("No data provided")
        else:
            # Use provided DataFrame
            mask = ~data.isna().any(axis=1)
            clean_df = data.loc[mask].reset_index(drop=True)

            self.df = clean_df
            
        # Extract Cell IDs for coloring
        if 'Cell ID' in self.df.columns:
            self.labels = self.df['Cell ID'].values
        else:
            self.labels = np.arange(len(self.df))
            
        # Extract feature columns (exclude Cell ID, Global X, Global Y, and columns with "N/A")
        exclude_cols = ['Cell ID', 'Global X', 'Global Y']
        self.feature_cols = [col for col in self.df.columns 
                            if col not in exclude_cols 
                            and not col.startswith('N/A')]
        self.data = self.df[self.feature_cols].values
        # scaler = StandardScaler(with_mean=False)
        
        # self.data = scaler.fit_transform(self.data)

        
        self.umap_model = None

        self.initUI()

    def initUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Info label
        self.info_label = QLabel(f"Data: {self.data.shape[0]} cells, {self.data.shape[1]} features")
        layout.addWidget(self.info_label)

        # Sliders and labels
        self.n_neighbors_label = QLabel("n_neighbors: 15")
        self.n_neighbors_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_neighbors_slider.setMinimum(2)
        self.n_neighbors_slider.setMaximum(1000)
        self.n_neighbors_slider.setValue(15)
        self.n_neighbors_slider.valueChanged.connect(self.update_n_neighbors_label)

        self.min_dist_label = QLabel("min_dist: 0.1")
        self.min_dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_dist_slider.setMinimum(1)
        self.min_dist_slider.setMaximum(100)
        self.min_dist_slider.setValue(10)
        self.min_dist_slider.valueChanged.connect(self.update_min_dist_label)

        self.n_epochs_label = QLabel("n_epochs: 200")
        self.n_epochs_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_epochs_slider.setMinimum(50)
        self.n_epochs_slider.setMaximum(1000)
        self.n_epochs_slider.setValue(200)
        self.n_epochs_slider.valueChanged.connect(self.update_n_epochs_label)

        layout.addWidget(self.n_neighbors_label)
        layout.addWidget(self.n_neighbors_slider)
        layout.addWidget(self.min_dist_label)
        layout.addWidget(self.min_dist_slider)
        layout.addWidget(self.n_epochs_label)
        layout.addWidget(self.n_epochs_slider)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Run UMAP")
        self.start_button.clicked.connect(self.run_umap)
        button_layout.addWidget(self.start_button)

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

    def update_n_neighbors_label(self):
        value = self.n_neighbors_slider.value()
        self.n_neighbors_label.setText(f"n_neighbors: {value}")

    def update_min_dist_label(self):
        value = self.min_dist_slider.value() / 100
        self.min_dist_label.setText(f"min_dist: {value:.2f}")

    def update_n_epochs_label(self):
        value = self.n_epochs_slider.value()
        self.n_epochs_label.setText(f"n_epochs: {value}")


    def run_umap(self):
        n_neighbors = self.n_neighbors_slider.value()
        min_dist = self.min_dist_slider.value() / 100
        n_epochs = self.n_epochs_slider.value()

        # Disable button during computation
        self.start_button.setEnabled(False)
        self.start_button.setText("Running UMAP...")
        QApplication.processEvents()

        # --- Run UMAP ---
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_state=42,
        )
        embedding = self.umap_model.fit_transform(self.data)

        # --- Hierarchical clustering ---
        # Compute hierarchical linkage
        Z = linkage(embedding, method='ward')

        # Option 1: Dynamically choose number of clusters using distance threshold
        # Smaller t → more clusters
        distance_threshold = np.percentile(Z[:, 2], 90)  # tune this for granularity
        # cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

        # Option 2 (alternative): Agglomerative clustering with automatic linkage distance
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        cluster_labels = clustering.fit_predict(embedding)

        # Compute cluster centroids
        unique_clusters = np.unique(cluster_labels)
        centroids = np.array([
            embedding[cluster_labels == c].mean(axis=0)
            for c in unique_clusters
        ])

        # Compute distance of each centroid from origin
        centroid_dists = np.linalg.norm(centroids, axis=1)
        cluster_color_map = {c: centroid_dists[i] for i, c in enumerate(unique_clusters)}

        # Assign color values based on centroid distances
        distance_colors = np.array([cluster_color_map[c] for c in cluster_labels])

        # --- Hover data ---
        hover_data = pd.DataFrame({
            'Cluster': cluster_labels
        })

        if 'Global X' in self.df.columns:
            hover_data['Global X'] = self.df['Global X'].values
        if 'Global Y' in self.df.columns:
            hover_data['Global Y'] = self.df['Global Y'].values

        for col in self.feature_cols:
            hover_data[col] = self.df[col].values

        # --- Interactive UMAP plot ---
        p = umap.plot.interactive(
            self.umap_model,
            labels=distance_colors,
            hover_data=hover_data,
            point_size=5,
            theme='viridis'
        )
        umap.plot.show(p)

        # Re-enable button
        self.start_button.setEnabled(True)
        self.start_button.setText("Run UMAP")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Example usage with custom DataFrame:
    df = pd.read_csv('KC25 4242_complete_cell_data.csv')
    window = UMAPVisualizer(data=df)
    
    # Or use default synthetic data:
    # window = UMAPVisualizer()
    window.show()
    sys.exit(app.exec())