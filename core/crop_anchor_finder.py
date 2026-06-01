"""Auto-find the crop anchor that maps the reference (decoding cycle 1) top-left
into the larger protein (moving) image.

The brightfield channel is a blob field (millions of small blobs) assembled from
slides, so SIFT/texture matching does not apply and per-blob constellation matching
is too slow over a 20k-image. Instead we treat the blurred, downsampled blob field
as a smooth low-frequency density texture and run a fast masked template match of a
small reference top-left patch over the protein image, sweeping rotation (scale is
assumed 1:1). The top-N correlation peaks become ranked candidates; the best few are
optionally sharpened with ORB + RANSAC. Each candidate carries a 2x3 affine transform
T (reference -> protein); the crop anchor is T applied to reference (0, 0).
"""

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from utils import calculate_ncc, to_uint8

logger = logging.getLogger(__name__)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Reduce an image to a single-channel 2D array without changing dtype."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] in (3, 4):
            return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        return img[..., 0]
    raise ValueError(f"Unsupported image shape {img.shape}")


def _density_map(gray_u8: np.ndarray, blur_k: int) -> np.ndarray:
    """Blur the blob field into a smooth density texture, mean-subtracted float32.

    Mean subtraction lets a masked TM_CCORR_NORMED behave like normalized
    cross-correlation (OpenCV only supports masks for CCORR_NORMED / SQDIFF).
    """
    k = max(3, int(blur_k) | 1)  # force odd, >= 3
    blurred = cv2.GaussianBlur(gray_u8, (k, k), 0).astype(np.float32)
    return blurred - float(blurred.mean())


def _rotate_expand(img: np.ndarray, angle: float):
    """Rotate ``img`` by ``angle`` (getRotationMatrix2D convention) onto an
    expanded canvas. Returns (rotated, mask, ox, oy) where (ox, oy) is where the
    original top-left pixel (0, 0) lands in the rotated image."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - center[0]
    M[1, 2] += nh / 2.0 - center[1]
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(
        np.full((h, w), 255, np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST
    )
    # reference-local origin (0, 0) -> M @ (0, 0, 1) = (M[0,2], M[1,2])
    return rotated, mask, float(M[0, 2]), float(M[1, 2])


class CropAnchorFinder(QThread):
    """Worker that proposes ranked crop-anchor candidates."""

    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    candidates_ready = pyqtSignal(list)

    def __init__(
        self,
        reference_img: np.ndarray,
        moving_img: np.ndarray,
        patch_size: int = 1500,
        num_candidates: int = 5,
        angle_range: float = 15.0,
        angle_step: float = 3.0,
        target_long_side: int = 2000,
        refine: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.reference_img = reference_img
        self.moving_img = moving_img
        self.patch_size = int(patch_size)
        self.num_candidates = max(1, int(num_candidates))
        self.angle_range = float(angle_range)
        self.angle_step = max(0.5, float(angle_step))
        self.target_long_side = int(target_long_side)
        self.refine = bool(refine)

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

        # Small top-left reference patch.
        s = min(self.patch_size, ref.shape[0], ref.shape[1])
        patch_full = ref[:s, :s]

        # Downscale the moving image (uint16-safe) so the search is cheap.
        ds = min(1.0, self.target_long_side / float(max(mov.shape[:2])))
        mov_small = cv2.resize(
            mov,
            (max(1, int(mov.shape[1] * ds)), max(1, int(mov.shape[0] * ds))),
            interpolation=cv2.INTER_AREA,
        )
        patch_small = cv2.resize(
            patch_full,
            (max(1, int(s * ds)), max(1, int(s * ds))),
            interpolation=cv2.INTER_AREA,
        )
        self.progress.emit(20, "Building density maps")

        blur_k = max(3, int(round(min(patch_small.shape[:2]) * 0.05)))
        mov_map = _density_map(to_uint8(mov_small), blur_k)
        patch_map = _density_map(to_uint8(patch_small), blur_k)

        ph, pw = patch_map.shape[:2]
        mh, mw = mov_map.shape[:2]
        if mh < ph or mw < pw:
            raise ValueError("Reference patch is larger than the moving image.")

        # Accumulate the best correlation (and angle) per ref-origin location.
        acc = np.full((mh, mw), -1.0, dtype=np.float32)
        ang_map = np.zeros((mh, mw), dtype=np.float32)

        angles = self._angle_sweep()
        for i, angle in enumerate(angles):
            rot, mask, ox, oy = _rotate_expand(patch_map, angle)
            rh, rw = rot.shape[:2]
            if mh < rh or mw < rw:
                continue
            res = cv2.matchTemplate(mov_map, rot, cv2.TM_CCORR_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            # res[ty, tx] -> ref-origin at (tx + ox, ty + oy)
            y0 = int(round(oy))
            x0 = int(round(ox))
            rh2, rw2 = res.shape[:2]
            y1 = min(mh, y0 + rh2)
            x1 = min(mw, x0 + rw2)
            if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
                continue
            sub = res[: y1 - y0, : x1 - x0]
            region = acc[y0:y1, x0:x1]
            better = sub > region
            region[better] = sub[better]
            ang_map[y0:y1, x0:x1][better] = angle
            self.progress.emit(
                20 + int(50 * (i + 1) / len(angles)), "Coarse rotation search"
            )

        radius = max(1, int(round(s * ds * 0.5)))
        peaks = self._nms_peaks(acc, self.num_candidates, radius)

        candidates = []
        for py, px, score in peaks:
            angle = float(ang_map[py, px])
            anchor = (px / ds, py / ds)
            T = self._transform_from(anchor, angle)
            candidates.append(
                {"anchor": anchor, "angle": angle, "score": float(score), "T": T}
            )

        if self.refine:
            self.progress.emit(75, "Refining top candidates")
            for cand in candidates:
                self._refine(cand, ref, mov, patch_full, s)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        self.progress.emit(100, "Done")
        return candidates

    # -- helpers -----------------------------------------------------------
    def _angle_sweep(self) -> list[float]:
        if self.angle_range <= 0:
            return [0.0]
        n = int(round(self.angle_range / self.angle_step))
        angles = sorted({round(k * self.angle_step, 3) for k in range(-n, n + 1)})
        if 0.0 not in angles:
            angles.append(0.0)
        return sorted(angles)

    @staticmethod
    def _transform_from(anchor, angle_deg: float) -> np.ndarray:
        """Build the 2x3 affine T (reference -> protein) for a rigid placement.

        Matches getRotationMatrix2D's convention: A = [[c, s], [-s, c]].
        """
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array(
            [[c, s, anchor[0]], [-s, c, anchor[1]]], dtype=np.float64
        )

    @staticmethod
    def _nms_peaks(acc: np.ndarray, n: int, radius: int) -> list[tuple]:
        work = acc.copy()
        peaks = []
        for _ in range(n):
            idx = int(np.argmax(work))
            py, px = np.unravel_index(idx, work.shape)
            score = float(work[py, px])
            if score <= 0:
                break
            peaks.append((int(py), int(px), score))
            y0 = max(0, py - radius)
            y1 = min(work.shape[0], py + radius + 1)
            x0 = max(0, px - radius)
            x1 = min(work.shape[1], px + radius + 1)
            work[y0:y1, x0:x1] = -1.0
        return peaks

    def _refine(self, cand: dict, ref, mov, patch_full, s) -> None:
        """Sharpen a coarse candidate with ORB + RANSAC partial-affine.

        Keeps the coarse estimate if matching is unreliable.
        """
        try:
            ax, ay = cand["anchor"]
            margin = int(0.15 * s)
            x0 = int(max(0, ax - margin))
            y0 = int(max(0, ay - margin))
            x1 = int(min(mov.shape[1], ax + s + margin))
            y1 = int(min(mov.shape[0], ay + s + margin))
            region = mov[y0:y1, x0:x1]
            if region.shape[0] < 16 or region.shape[1] < 16:
                return

            patch_u8 = to_uint8(patch_full)
            region_u8 = to_uint8(region)
            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(patch_u8, None)
            kp2, des2 = orb.detectAndCompute(region_u8, None)
            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                return
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            knn = matcher.knnMatch(des1, des2, k=2)
            good = [
                m
                for pair in knn
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            if len(good) < 8:
                return
            # reference-local == reference-absolute (patch is the top-left).
            src = np.float32([kp1[m.queryIdx].pt for m in good])
            dst = np.float32(
                [[kp2[m.trainIdx].pt[0] + x0, kp2[m.trainIdx].pt[1] + y0] for m in good]
            )
            M, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
            if M is None or inliers is None or int(inliers.sum()) < 8:
                return

            # Score by NCC: derotate the protein region into the reference frame
            # (output sized like the patch) and compare against the patch. Built
            # from the small region only, never the full moving image.
            inv = self._invert(M)
            inv_a = inv[:, :2]
            inv_t = inv_a @ np.array([x0, y0], dtype=np.float64) + inv[:, 2]
            region_inv = np.hstack([inv_a, inv_t.reshape(2, 1)]).astype(np.float32)
            warped = cv2.warpAffine(region_u8, region_inv, (s, s))
            ncc = calculate_ncc(warped, patch_u8)
            if ncc is None or ncc <= cand["score"]:
                return
            cand["T"] = M.astype(np.float64)
            cand["anchor"] = (float(M[0, 2]), float(M[1, 2]))
            cand["angle"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            cand["score"] = float(ncc)
        except Exception as exc:  # pragma: no cover - refinement is best-effort
            logger.debug("Refinement skipped: %s", exc)

    @staticmethod
    def _invert(T: np.ndarray) -> np.ndarray:
        A = np.asarray(T, dtype=np.float64)[:, :2]
        t = np.asarray(T, dtype=np.float64)[:, 2]
        invA = np.linalg.inv(A)
        return np.hstack([invA, (-invA @ t).reshape(2, 1)])
