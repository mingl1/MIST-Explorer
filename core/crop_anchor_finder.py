"""Auto-find the crop anchor that maps the reference (decoding cycle 1) top-left
into the larger protein (moving) image.

Matching is done with ORB features at full resolution (no downscaling, no density
map) between the contrast-adjusted reference top-left patch and the contrast-adjusted
moving image. ORB is rotation-robust and is already used in the alignment pipeline.
RANSAC partial-affine fits a transform; running it repeatedly on the remaining
(non-inlier) matches yields ranked candidates. Each candidate carries a 2x3 affine
transform T (reference -> protein); the crop anchor is T applied to reference (0, 0).
"""

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.spatial import cKDTree

from utils import adjust_contrast, calculate_ncc

logger = logging.getLogger(__name__)

_MIN_INLIERS = 8
_EMPTY_PTS = np.empty((0, 2), dtype=np.float32)
_BLOB_MATCH_RADIUS = 6.0


def _blob_centroids(gray_u8: np.ndarray, max_pts: int = 4000) -> np.ndarray:
    """Centroids of bright blobs (adaptive threshold + contour moments)."""
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 0:
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    if not pts:
        return _EMPTY_PTS
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) > max_pts:
        idx = np.random.default_rng(0).choice(len(arr), max_pts, replace=False)
        arr = arr[idx]
    return arr


def _matched_blob_fraction(
    ref_pts: np.ndarray, region_pts: np.ndarray, radius: float = _BLOB_MATCH_RADIUS
) -> float:
    """Fraction of reference blobs with a region blob within ``radius`` px.

    Measures whether the blob *constellation* lines up after warping -- the most
    alignment-relevant signal for a blob field.
    """
    if len(ref_pts) == 0 or len(region_pts) == 0:
        return 0.0
    tree = cKDTree(region_pts)
    dist, _ = tree.query(ref_pts, k=1, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(dist)))


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Reduce an image to a single-channel 2D array without changing dtype."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] in (3, 4):
            return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        return img[..., 0]
    raise ValueError(f"Unsupported image shape {img.shape}")


def contrast_to_uint8(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile contrast-adjust an image and return uint8 (for ORB / display)."""
    gray = _to_gray(img).astype(np.float32, copy=False)
    adjusted = adjust_contrast(gray, low, high)  # float32 in [0, 1]
    return (adjusted * 255.0).astype(np.uint8)


class CropAnchorFinder(QThread):
    """Worker that proposes ranked crop-anchor candidates via ORB matching."""

    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    candidates_ready = pyqtSignal(list)

    def __init__(
        self,
        reference_img: np.ndarray,
        moving_img: np.ndarray,
        patch_size: int = 2000,
        num_candidates: int = 5,
        n_features: int = 50000,
        ratio: float = 0.75,
        ransac_thresh: float = 8.0,
        parent=None,
    ):
        super().__init__(parent)
        self.reference_img = reference_img
        self.moving_img = moving_img
        self.patch_size = int(patch_size)
        self.num_candidates = max(1, int(num_candidates))
        self.n_features = int(n_features)
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)

    # -- public ------------------------------------------------------------
    def run(self):
        try:
            candidates = self.find_candidates()
            self.candidates_ready.emit(candidates)
        except Exception as exc:  # pragma: no cover - signal path
            logger.error("CropAnchorFinder failed: %s", exc, exc_info=True)
            self.error.emit(str(exc))

    def find_candidates(self) -> list[dict]:
        ref = _to_gray(self.reference_img)
        mov = _to_gray(self.moving_img)

        # Small top-left reference patch, contrast-adjusted to uint8.
        s = min(self.patch_size, ref.shape[0], ref.shape[1])
        patch = contrast_to_uint8(ref[:s, :s])
        self.progress.emit(15, "Contrast-adjusting moving image")
        mov_u8 = contrast_to_uint8(mov)

        orb = cv2.ORB_create(nfeatures=self.n_features)
        self.progress.emit(30, "Detecting reference features")
        kp1, des1 = orb.detectAndCompute(patch, None)
        self.progress.emit(45, "Detecting moving features")
        kp2, des2 = orb.detectAndCompute(mov_u8, None)
        if des1 is None or des2 is None or len(kp1) < _MIN_INLIERS or len(kp2) < _MIN_INLIERS:
            logger.warning("Not enough ORB features for matching.")
            return []

        self.progress.emit(60, "Matching features")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des1, des2, k=2)
        good = [
            m
            for pair in knn
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.ratio * n.distance
        ]
        if len(good) < _MIN_INLIERS:
            logger.warning("Too few good matches (%d).", len(good))
            return []

        # patch coords == reference-absolute coords (patch is the top-left).
        src_all = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_all = np.float32([kp2[m.trainIdx].pt for m in good])

        candidates = []
        remaining = list(range(len(good)))
        for _ in range(self.num_candidates):
            if len(remaining) < _MIN_INLIERS:
                break
            src = src_all[remaining]
            dst = dst_all[remaining]
            M, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=10000,
                confidence=0.99,
            )
            if M is None or inliers is None:
                break
            inl = inliers.flatten().astype(bool)
            n_inl = int(inl.sum())
            if n_inl < _MIN_INLIERS:
                break

            T = M.astype(np.float64)
            inl_src = src[inl]
            inl_dst = dst[inl]
            proj = (T[:, :2] @ inl_src.T).T + T[:, 2]
            residual = float(np.mean(np.linalg.norm(proj - inl_dst, axis=1)))
            spread = float(np.hypot(inl_src[:, 0].std(), inl_src[:, 1].std()))
            candidates.append(
                {
                    "anchor": (float(T[0, 2]), float(T[1, 2])),
                    "angle": float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))),
                    "inliers": n_inl,
                    "inlier_ratio": float(n_inl / len(remaining)),
                    "residual": residual,
                    "spread": spread,
                    "blob_fraction": 0.0,  # filled below
                    "score": float(n_inl),  # refined to NCC below
                    "T": T,
                }
            )
            remaining = [remaining[i] for i in range(len(remaining)) if not inl[i]]

        # Photometric (NCC) + constellation (matched-blob fraction) scoring of the
        # derotated protein region against the reference patch.
        self.progress.emit(85, "Scoring candidates")
        ref_pts = _blob_centroids(patch)
        for cand in candidates:
            warped = self._derotate(cand["T"], mov_u8, s)
            ncc = calculate_ncc(warped, patch) if warped is not None else None
            cand["score"] = float(ncc) if ncc is not None else -1.0
            region_pts = _blob_centroids(warped) if warped is not None else _EMPTY_PTS
            cand["blob_fraction"] = _matched_blob_fraction(ref_pts, region_pts)

        candidates.sort(key=lambda c: (c["score"], c["inliers"]), reverse=True)
        self.progress.emit(100, "Done")
        return candidates

    # -- helpers -----------------------------------------------------------
    def _derotate(self, T, mov_u8, s):
        """Warp the protein image into the reference frame (output s x s)."""
        try:
            inv = self._invert(T).astype(np.float32)
            return cv2.warpAffine(mov_u8, inv, (s, s))
        except Exception:  # pragma: no cover - best-effort
            return None

    @staticmethod
    def _invert(T: np.ndarray) -> np.ndarray:
        A = np.asarray(T, dtype=np.float64)[:, :2]
        t = np.asarray(T, dtype=np.float64)[:, 2]
        invA = np.linalg.inv(A)
        return np.hstack([invA, (-invA @ t).reshape(2, 1)])
