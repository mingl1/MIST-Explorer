Analysis
========

The Analysis tab provides advanced statistical and visual analysis tools for exploring protein expression data within specific regions of interest. This interface allows researchers to select areas of the image, analyze protein expression patterns, and generate a variety of visualizations to gain deeper insights.

.. note::
   To include the analysis tab image in this documentation, save a screenshot of the analysis tab to the docs/source/_static directory with the filename "analysis_tab.png".

.. image:: _static/analysis_tab.png
   :width: 600
   :alt: Analysis tab interface

Interface Overview
-----------------

The Analysis tab consists of two main sections:

1. **Navigation Controls**: Buttons for navigating between different regions of interest and saving plots
2. **Content Area**: A scrollable area displaying the current region analysis and associated visualizations

Region of Interest (ROI) Management
----------------------------------

The Analysis tab allows you to define and analyze specific regions within your image. Each region becomes a separate analysis view that you can navigate between.

Region Selection Tools
^^^^^^^^^^^^^^^^^^^^^

The View tab provides selection tools that integrate with the Analysis tab:

* **Rectangle Selection**: Select rectangular regions for analysis
* **Circle/Ellipse Selection**: Select circular/oval regions for analysis
* **Polygon Selection**: Draw custom shapes to isolate complex structures

Data Selection
^^^^^^^^^^^^^

For each region of interest, you can select which proteins to include in your analysis:

* **Protein Selection Dropdown**: Multi-select dropdown that allows you to choose specific proteins for analysis
* **Select All/Deselect All**: Quickly select or deselect all proteins
* **Apply Button**: Update visualizations based on your protein selection

Navigation Between Regions
-------------------------

The Analysis tab provides intuitive navigation controls for working with multiple regions:

* **Back Button**: Navigate to the previous region of interest
* **Next Button**: Navigate to the next region of interest
* **Save Plot**: Save the current visualization as an image file

Each region of interest shows:
* Region bounds information
* Color indicator representing the region selection
* Delete button to remove the current region

Visualization Options
-------------------

The Analysis tab provides multiple visualization types for analyzing protein expression data:

Boxplot
^^^^^^^

Displays the distribution of protein expression levels within the selected region. This visualization shows:

* Median expression level (center line)
* Interquartile range (box)
* Outliers (points)
* Range of non-outlier data (whiskers)

Z-Scores Heatmap
^^^^^^^^^^^^^^^

Visualizes standardized expression values (z-scores) across proteins and cells within the selected region. This helps identify proteins with unusually high or low expression.

Spatial Heatmap
^^^^^^^^^^^^^^

Maps the spatial distribution of protein expression within the selected region, helping to identify localized expression patterns and hotspots.

Pie Chart
^^^^^^^^

Shows the relative proportion of cells expressing different proteins within the selected region, helping to understand the cellular composition.

Histogram
^^^^^^^^

Displays the frequency distribution of expression values for selected proteins, revealing expression patterns and populations.

UMAP
^^^^

Implements Uniform Manifold Approximation and Projection for dimensionality reduction, allowing visualization of high-dimensional protein expression data in a 2D plot. This can reveal clusters of cells with similar expression profiles.

Working with Visualizations
--------------------------

Each visualization offers additional capabilities:

* **Expand to New Window**: Open any visualization in a separate window for detailed viewing
* **Return to Graph List**: Go back to the main list of available visualizations
* **Add New Charts**: Create additional custom visualizations

Data Analysis Workflow
--------------------

A typical workflow for using the Analysis tab includes:

1. **Select Region**: Use the selection tools in the View tab to define a region of interest
2. **Choose Proteins**: Select which proteins to include in your analysis
3. **Explore Visualizations**: Navigate through the different visualization types
4. **Compare Regions**: Use Back and Next buttons to compare analyses across different regions
5. **Save Results**: Save important visualizations for reporting or further analysis

Technical Details
----------------

The Analysis tab implements several advanced techniques for data analysis:

1. **Region Filtering**: Implements geometric algorithms to precisely extract cell data from defined regions:
   * Rectangle filtering using coordinate bounds
   * Circle/oval filtering using elliptical equations
   * Polygon filtering using ray casting algorithm

2. **Lazy Loading**: Visualizations are generated on-demand to improve performance with large datasets.

3. **Interactive Visualizations**: Matplotlib integration provides high-quality, interactive scientific visualizations.

4. **Multi-window Support**: The ability to pop out visualizations into separate windows allows for detailed examination and side-by-side comparison.

Usage Tips
---------

* **Strategic Region Selection**: Select regions that contain interesting features or contrasting cell populations
* **Protein Comparison**: Select related proteins to compare their expression patterns within the same region
* **Save Visualizations**: Use the Save Plot button to capture important findings for publications or presentations
* **Multiple Windows**: Use the expand feature to compare multiple visualizations side by side
* **Region Navigation**: Create multiple regions to systematically analyze different areas of your image
