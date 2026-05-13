"""
Status badge delegate for the ImageManager tree.

Draws small colored pill badges on tree items to indicate which role each
image/channel currently holds (canvas, reference, segmentation source/label,
tissue target/moving).  The delegate is stateless: it queries ImageStorage
at paint time so it always reflects current state without any mirroring.
"""

from enum import Enum

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from core import ImageStorage
from core.project_naming import is_segmentation_channel

_BADGE_HEIGHT = 14
_BADGE_PADDING_X = 6
_BADGE_SPACING = 3
_FONT_SIZE = 7


class BadgeStyle(Enum):
    """Visual style for each role badge: (display label, bg_key, fg_key)."""

    VIEW = ("View", "badge_view_bg", "badge_fg")
    REF = ("Ref", "badge_ref_bg", "badge_fg")
    LABEL = ("Label", "badge_label_bg", "badge_fg")

    def __init__(self, label: str, bg_key: str, fg_key: str):
        self.label = label
        self._bg_key = bg_key
        self._fg_key = fg_key

    @property
    def bg(self) -> str:
        try:
            from ui.theme import ThemeManager
            return ThemeManager.instance().get(self._bg_key)
        except RuntimeError:
            return {"badge_view_bg": "#3b82f6", "badge_ref_bg": "#22c55e", "badge_label_bg": "#a855f7"}[self._bg_key]

    @property
    def fg(self) -> str:
        try:
            from ui.theme import ThemeManager
            return ThemeManager.instance().get(self._fg_key)
        except RuntimeError:
            return "#ffffff"


# Stable left-to-right display order for badges
_BADGE_ORDER: list[BadgeStyle] = [
    BadgeStyle.VIEW,
    BadgeStyle.REF,
    BadgeStyle.LABEL,
]

# Maps ImageStorage key → BadgeStyle for UUID-level role checks
_STORAGE_ROLE_MAP: list[tuple[str, BadgeStyle]] = [
    ("canvas_uuid", BadgeStyle.VIEW),
    ("reference_uuid", BadgeStyle.REF),
    ("seg_label_uuid", BadgeStyle.LABEL),
]


def _resolve_stored_channel(
    storage: ImageStorage, storage_key: str, entry: dict
) -> str | None:
    """Return the stored channel as a 'Channel N' string for the given role entry.

    Returns None for roles that carry no channel (VIEW / canvas_uuid), which means
    the badge should only appear on root items, never on channel children.
    """
    if storage_key == "reference_uuid":
        ref_ch = storage.get_data("reference_channel")
        return ref_ch["value"] if ref_ch else None
    if storage_key == "seg_label_uuid":
        ch = entry.get("channel")
        return f"Channel {ch + 1}" if isinstance(ch, int) else None
    if storage_key == "canvas_uuid":
        canvas_ch = storage.get_data("canvas_channel")
        return canvas_ch["value"] if canvas_ch else None
    return None


class StatusBadgeDelegate(QStyledItemDelegate):
    """
    Draws colored pill badges on the right side of each ImageManager tree item
    to indicate which role that image/channel currently holds.
    """

    def _compute_badges(
        self, uuid_str: str, channel_name: str | None, is_root: bool
    ) -> frozenset[BadgeStyle]:
        storage = ImageStorage()
        badges: set[BadgeStyle] = set()

        for storage_key, badge_style in _STORAGE_ROLE_MAP:
            entry = storage.get_data(storage_key)
            if not entry or str(entry["value"]) != uuid_str:
                continue

            if is_root:
                badges.add(badge_style)
            else:
                # Channel child: only badge the specific matching channel.
                # Roles with no stored channel (VIEW) are root-only.
                stored_ch = _resolve_stored_channel(storage, storage_key, entry)
                if stored_ch is not None and stored_ch == channel_name:
                    badges.add(badge_style)

        # Label: any channel whose wrapper carries a segmentation name (channel child only)
        if not is_root:
            img = storage.get_data(uuid_str)
            if img and channel_name:
                wrapper = img.get("data", {}).get(channel_name)
                if wrapper is not None and is_segmentation_channel(wrapper):
                    badges.add(BadgeStyle.LABEL)

        return frozenset(badges)

    def initStyleOption(self, option: QStyleOptionViewItem, index):
        super().initStyleOption(option, index)
        # Root items use 50 px icons; channel children use 30 px — tell Qt to
        # allocate only the space actually needed so child rows have no gap.
        is_root = not index.parent().isValid()
        icon_px = 50 if is_root else 30
        option.decorationSize = QSize(icon_px, icon_px)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        super().paint(painter, option, index)

        uuid_val = index.data(Qt.ItemDataRole.UserRole)
        channel_name = index.data(Qt.ItemDataRole.WhatsThisRole)

        if uuid_val is None:
            return

        uuid_str = str(uuid_val)
        is_root = not index.parent().isValid()
        badges = self._compute_badges(uuid_str, channel_name, is_root)
        if not badges:
            return

        active = [b for b in _BADGE_ORDER if b in badges]
        if not active:
            return

        painter.save()
        font = painter.font()
        font.setPointSize(_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        x = option.rect.right() - 2
        y = option.rect.center().y() - _BADGE_HEIGHT // 2

        for badge in reversed(active):
            text_w = fm.horizontalAdvance(badge.label)
            badge_w = text_w + _BADGE_PADDING_X * 2
            x -= badge_w
            rect = QRectF(x, y, badge_w, _BADGE_HEIGHT)

            path = QPainterPath()
            path.addRoundedRect(rect, 3, 3)
            painter.fillPath(path, QColor(badge.bg))
            painter.setPen(QColor(badge.fg))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge.label)

            x -= _BADGE_SPACING

        painter.restore()
