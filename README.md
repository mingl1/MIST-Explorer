# MIST-Explorer

Our application is a powerful and intuitive tool designed for researchers and scientists in the field of single-cell proteomics. It allows users to load, visualize, and analyze protein distribution at the single-cell level. By providing a user-friendly interface, the app enables seamless exploration of protein expression data, helping to uncover insights into cellular functions, interactions, and heterogeneity.

## Key Features
- **Basic Image Manipulation**: Efficiently crop, rotate, flip tiff images. 
- **Multi-Channel Viewer**: Layer-by-layer visualization with independent opacity, contrast, and color controls
- **Image Registration**: Multi-algorithm alignment (Optical Flow, SIFT/ORB, ITK B-spline) with sub to few-pixel accuracy
- **Cell Segmentation**: StarDist neural network integration for accurate single-cell detection
- **Single-Cell Analysis**: Extract protein expression levels with bead decoding and ROI-based statistics
- **UMAP & Interactive Clustering**: Dimensionality reduction and Leiden clustering for cell type identification
- **Interactive ROI Tools**: Rectangular and circular region selection for focused analysis

## Advanced Analytics
- **Statistical Analysis**: Distribution analysis, histograms, and IQR-based outlier detection
- **Export Capabilities**: Save results and visualizations for external processing

## Requirements
- **Operating System**: Windows, macOS tested 
- **Minimum Hardware**: 8 GB RAM, 2 GHz Processor
- **Dependencies**: Python (with libraries such as NumPy, Pandas, Matplotlib), JavaScript (for web-based interfaces), and cloud-based storage options (if applicable).

## Installation
1. Install [uv](https://docs.astral.sh/uv/):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone and setup:
   ```
   git clone https://github.com/yourusername/protein_visualization_app.git
   cd MIST-Explorer
   uv sync
   ```

## Usage
```
uv run main.py
```
