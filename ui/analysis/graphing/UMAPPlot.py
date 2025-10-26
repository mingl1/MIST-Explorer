import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, 
    QHBoxLayout, QComboBox, QDoubleSpinBox, QGroupBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import clr
from kneed import KneeLocator

class UMAPVisualizer(QMainWindow):
    def __init__(self, data=None):
        super().__init__()

        self.setWindowTitle("UMAP Visualizer")
        self.setGeometry(100, 100, 1000, 800)
        
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
        
        # Preprocess data
        self.data = self.df[self.feature_cols].values
        # raw_data[raw_data == 0] = 1
        # transformed_data = np.array(clr(raw_data))
        # scaler = StandardScaler()
        # self.data = scaler.fit_transform(transformed_data)
        
        # Create AnnData object
        self.adata = sc.AnnData(X=self.data)
        self.adata.obs['cell_id'] = self.labels
        self.adata.var_names = self.feature_cols
        self.adata.layers["counts"] = self.adata.X.copy()
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)



        
        # Pre-compute PCA with maximum components for dynamic selection
        max_comps = min(100, self.data.shape[0] - 1, self.data.shape[1])
        sc.tl.pca(self.adata, n_comps=max_comps-1, svd_solver='arpack')
        self.max_pca_components = max_comps
        
        self.initUI()

    def initUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Info label
        self.info_label = QLabel(f"Data: {self.data.shape[0]} cells, {self.data.shape[1]} features")
        layout.addWidget(self.info_label)

        # PCA Selection Group
        pca_group = QGroupBox("PCA Component Selection")
        umap_group = QGroupBox("UMAP COnfig")
        pca_layout = QVBoxLayout()
        umap_layout = QVBoxLayout()
        
        # PCA method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("PCA Method:"))
        self.pca_method_combo = QComboBox()
        self.pca_method_combo.addItems(["Manual", "Variance Threshold", "Elbow Method"])
        self.pca_method_combo.currentTextChanged.connect(self.on_pca_method_changed)
        method_layout.addWidget(self.pca_method_combo)
        method_layout.addStretch()
        pca_layout.addLayout(method_layout)
        
        # Manual slider (default visible)
        self.manual_widget = QWidget()
        manual_layout = QVBoxLayout()
        manual_layout.setContentsMargins(0, 0, 0, 0)
        self.n_components_label = QLabel("PCA n_components: 50")
        self.n_components_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_components_slider.setMinimum(10)
        self.n_components_slider.setMaximum(self.max_pca_components)
        self.n_components_slider.setValue(min(50, self.max_pca_components))
        self.n_components_slider.valueChanged.connect(self.update_n_components_label)
        manual_layout.addWidget(self.n_components_label)
        manual_layout.addWidget(self.n_components_slider)
        self.manual_widget.setLayout(manual_layout)
        pca_layout.addWidget(self.manual_widget)
        
        # Variance threshold selector (hidden by default)
        self.variance_widget = QWidget()
        variance_layout = QHBoxLayout()
        variance_layout.setContentsMargins(0, 0, 0, 0)
        variance_layout.addWidget(QLabel("Variance Threshold:"))
        self.variance_spinbox = QDoubleSpinBox()
        self.variance_spinbox.setRange(0.5, 0.99)
        self.variance_spinbox.setSingleStep(0.05)
        self.variance_spinbox.setValue(0.85)
        self.variance_spinbox.setSuffix("%")
        self.variance_spinbox.setDecimals(2)
        variance_layout.addWidget(self.variance_spinbox)
        self.variance_result_label = QLabel("")
        variance_layout.addWidget(self.variance_result_label)
        variance_layout.addStretch()
        self.variance_widget.setLayout(variance_layout)
        self.variance_widget.hide()
        pca_layout.addWidget(self.variance_widget)
        
        # Elbow method info (hidden by default)
        self.elbow_widget = QWidget()
        elbow_layout = QVBoxLayout()
        elbow_layout.setContentsMargins(0, 0, 0, 0)
        self.elbow_result_label = QLabel("Elbow will be computed automatically")
        elbow_layout.addWidget(self.elbow_result_label)
        self.elbow_widget.setLayout(elbow_layout)
        self.elbow_widget.hide()
        pca_layout.addWidget(self.elbow_widget)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group)

        # Sliders and labels
        self.n_neighbors_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_neighbors_slider.setMinimum(2)
        self.n_neighbors_slider.setMaximum(200)
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
        self.n_neighbors_label = QLabel("n_labels: 15")
        umap_layout.addWidget(self.n_neighbors_label)
        umap_layout.addWidget(self.n_neighbors_slider)
        umap_layout.addWidget(self.min_dist_label)
        umap_layout.addWidget(self.min_dist_slider)
        umap_layout.addWidget(self.n_epochs_label)
        umap_layout.addWidget(self.n_epochs_slider)
        
        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Run UMAP")
        self.start_button.clicked.connect(self.run_umap)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

        # Matplotlib canvas for plotting
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        central_widget.setLayout(layout)

    def on_pca_method_changed(self, method):
        """Show/hide appropriate widgets based on PCA method selection"""
        self.manual_widget.setVisible(method == "Manual")
        self.variance_widget.setVisible(method == "Variance Threshold")
        self.elbow_widget.setVisible(method == "Elbow Method")
        
    def compute_pca_components(self):
        """Compute number of PCA components based on selected method"""
        method = self.pca_method_combo.currentText()
        
        if method == "Manual":
            return self.n_components_slider.value()
        
        elif method == "Variance Threshold":
            threshold = self.variance_spinbox.value() / 100  # Convert percentage to decimal
            var_ratios = self.adata.uns['pca']['variance_ratio']
            cumsum_var = np.cumsum(var_ratios)
            n_comps = np.argmax(cumsum_var >= threshold) + 1
            self.variance_result_label.setText(
                f"→ {n_comps} components ({cumsum_var[n_comps-1]:.2%} variance)"
            )
            return n_comps
        
        elif method == "Elbow Method":
            var_ratios = self.adata.uns['pca']['variance_ratio']
            x = np.arange(len(var_ratios))
            
            # Use kneedle algorithm to find elbow
            kneedle = KneeLocator(
                x, var_ratios, 
                curve='convex', 
                direction='decreasing',
                S=1.0
            )
            
            if kneedle.elbow is not None:
                n_comps = kneedle.elbow + 1
            else:
                # Fallback: use 90% variance if elbow not found
                cumsum_var = np.cumsum(var_ratios)
                n_comps = np.argmax(cumsum_var >= 0.90) + 1
            
            cumsum_var = np.cumsum(var_ratios)
            self.elbow_result_label.setText(
                f"Elbow detected at {n_comps} components ({cumsum_var[n_comps-1]:.2%} variance)"
            )
            return n_comps
        
        return 50  # Default fallback

    def update_n_components_label(self):
        value = self.n_components_slider.value()
        self.n_components_label.setText(f"PCA n_components: {value}")

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
        # Get PCA components based on selected method
        n_components = self.compute_pca_components()
        
        # Get other parameters from sliders
        n_neighbors = self.n_neighbors_slider.value()
        min_dist = self.min_dist_slider.value() / 100
        n_epochs = self.n_epochs_slider.value()

        # Disable button during computation
        self.start_button.setEnabled(False)
        self.start_button.setText("Running PCA + UMAP...")
        QApplication.processEvents()

        try:
            # Use pre-computed PCA, but subset to n_components
            self.adata.obsm['X_pca_subset'] = self.adata.obsm['X_pca'][:, :n_components]
            
            # Compute neighbors with specified n_neighbors (using PCA representation)
            
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep='X_pca_subset')
            
            # Compute UMAP with specified parameters
            sc.tl.umap(
                self.adata, 
                # min_dist=min_dist,
                n_components=2,
                # maxiter=n_epochs,
            )
            # Run Leiden clustering
            sc.tl.leiden(self.adata, flavor="igraph", n_iterations=2)

            # Clear previous plot
            self.figure.clear()
            
            # Create subplots: UMAP and variance explained
            ax1 = self.figure.add_subplot(111)
            # ax2 = self.figure.add_subplot(122)
            
            # Plot UMAP
            sc.pl.umap(
                self.adata,
                color="leiden",
                size=20,
                ax=ax1,
                show=False
            )
            
            method = self.pca_method_combo.currentText()
            ax1.set_title(f'UMAP\n(PCA: {method}, {n_components} comps)')
            
            # Plot variance explained
            var_ratios = self.adata.uns['pca']['variance_ratio']
            cumsum_var = np.cumsum(var_ratios)
            
            # ax2.plot(range(1, len(var_ratios) + 1), cumsum_var, 'b-', linewidth=2)
            # ax2.axvline(x=n_components, color='r', linestyle='--', label=f'Selected: {n_components}')
            # ax2.axhline(y=cumsum_var[n_components-1], color='r', linestyle=':', alpha=0.5)
            # ax2.set_xlabel('Number of Components')
            # ax2.set_ylabel('Cumulative Variance Explained')
            # ax2.set_title('PCA Variance Explained')
            # ax2.grid(True, alpha=0.3)
            # ax2.legend()
            # ax2.set_xlim(0, min(100, len(var_ratios)))
            # ax2.set_ylim(0, 1)
            
            # Refresh canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Calculate variance explained
            var_explained = cumsum_var[n_components-1]
            
            self.info_label.setText(
                f"Data: {self.data.shape[0]} cells, {self.data.shape[1]} features | "
                f"PCA: {n_components} components ({var_explained:.2%} variance) | "
                f"Clusters: {len(self.adata.obs['leiden'].unique())}"
            )

        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")
            print(f"Error during computation: {e}")
            import traceback
            traceback.print_exc()

        # Re-enable button
        self.start_button.setEnabled(True)
        self.start_button.setText("Run UMAP")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Example usage with custom DataFrame:
    df = pd.read_csv('KC25 4242_complete_cell_data.csv')
    window = UMAPVisualizer(data=df)
    
    window.show()
    sys.exit(app.exec())