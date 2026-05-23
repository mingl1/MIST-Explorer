Extract Tab
===========

The Extract tab handles all image preparation and processing steps before visualization and analysis. It is organized into four sub-tabs: **Transform**, **Alignment**, **Segmentation**, and **Quantification**.

.. note::
   Save a screenshot of the Extract tab to ``docs/source/_static/extract_tab.png`` to display it here.

.. image:: _static/extract_tab.png
   :width: 600
   :alt: Extract tab interface

Transform
---------

The Transform sub-tab provides tools for basic image manipulation and multi-cycle image registration.

Crop
^^^^

Select a rectangular region to keep from the current image:

* **Image Selection Tool**: Click the image icon to draw a crop region on the canvas.
* **Clear Selection**: Click the X icon to discard the current selection.

Rotate
^^^^^^

Adjust the rotation angle using the slider, then click **Ok** to apply.

Flip
^^^^

Mirror the image along either axis:

* **Flip Horizontal**: Mirror the image left-to-right.
* **Flip Vertical**: Mirror the image top-to-bottom.

Manual Alignment
^^^^^^^^^^^^^^^^

Manually translate images to remove gross positional offsets before running automatic registration. This step is optional but strongly recommended — it gives the registration algorithm a better starting point and significantly improves accuracy.

After automatic registration completes, a **post-registration modal** displays the result. You can manually fine-tune the output in this modal before confirming.

Automatic Registration
^^^^^^^^^^^^^^^^^^^^^^

Precisely aligns multi-cycle images using a feature-based registration pipeline.

.. dropdown:: How it works

   The pipeline uses **optical flow** with **ORB (Oriented FAST and Rotated BRIEF)** feature detection as the primary method. If ORB cannot find sufficient correspondences (e.g. low-texture regions), **SIFT (Scale-Invariant Feature Transform)** is used as a fallback.

   To handle large images efficiently, the image is divided into overlapping tiles. Each tile is registered independently and the results are composited. The tiled approach accounts for local deformations — such as stage drift or sample distortion — that a single global transform would miss.

Alignment
---------

The Alignment sub-tab aligns the brightfield (transmitted light) layer to the fluorescence channels. This is separate from multi-cycle registration in the Transform sub-tab.

Settings
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - Max Size
     - 23000 px
     - Maximum image dimension during processing. Images larger than this are downsampled first to reduce memory usage.
   * - Number of Tiles
     - 5
     - How many tiles the image is divided into for alignment. More tiles improves accuracy for images with local deformations but increases processing time.
   * - Overlap
     - 500 px
     - Pixel overlap between adjacent tiles. Sufficient overlap ensures seamless stitching of tile alignment results.

Click **Run** to perform the alignment, or **Cancel** to abort.

.. dropdown:: How it works

   Brightfield-to-fluorescence alignment uses the same optical flow pipeline as multi-cycle registration: ORB feature detection with SIFT fallback, applied tile-by-tile. The overlapping tile strategy is critical for large images where a single global homography cannot capture local stage drift or sample distortion.

Segmentation
------------

The Segmentation sub-tab identifies individual cells in your images. The resulting segmentation masks are used by the View and Analysis tabs to associate pixel intensities with specific cells.

Gaussian Blur (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^

A pre-processing blur that reduces high-frequency noise before segmentation. In most cases this is **not needed** — only apply it if significant noise is causing false detections.

StarDist
^^^^^^^^

StarDist uses a pre-trained convolutional neural network to detect and outline individual cells using star-convex polygon representations.

**Select Channel**: Choose the image channel to use for cell detection. A channel that clearly shows cell boundaries or nuclei gives the best results.

**Pre-trained 2D Model**: The default ``2D_versatile_fluo`` model works well for most fluorescence microscopy images.

**Parameters**:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - Percentile Low
     - 3.00
     - Lower percentile used for image intensity normalization before segmentation.
   * - Percentile High
     - 99.80
     - Upper percentile used for intensity normalization. Decrease if bright artifacts are suppressing cell signal.
   * - Probability Threshold
     - 0.48
     - Confidence threshold for accepting a cell detection. Increase to reduce false positives; decrease to capture more cells at the cost of more false positives.
   * - Overlap Threshold
     - 0.30
     - Maximum allowed IoU overlap between adjacent cell predictions. Lower values are more permissive about overlapping cells.

For a full explanation of the model architecture and parameter tuning guidance, see the `StarDist documentation <https://github.com/stardist/stardist>`_.

CellProfiler
^^^^^^^^^^^^

An alternative segmentation pipeline suited for larger regions of interest or cell types where StarDist performs poorly. Configure and run a CellProfiler pipeline, then load the resulting segmentation masks into MIST-Explorer.

Quantification
--------------

The Quantification sub-tab extracts per-cell protein intensity values from your segmented images.

Cell Intensity Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^

For each cell identified by segmentation, the pipeline computes the mean fluorescence intensity of each protein channel within the cell mask.

**Output**: A CSV file with one row per cell and one column per protein channel, plus spatial metadata (cell centroid coordinates). This file is used as input for the View and Analysis tabs.

.. dropdown:: How it works

   For each segmented cell mask, the pixel intensities of every protein channel are sampled and aggregated (mean by default). The cell centroid and bounding box are computed from the mask geometry. The result is a cell × protein expression matrix — the primary data structure used by the View and Analysis tabs.
