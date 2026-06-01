#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    files = [fn for fn in os.listdir(folder) if fn.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_mask(path):
    with Image.open(path) as im:
        return np.array(im)


def build_mask_path(mask_dir, img_filename):
    base = os.path.splitext(os.path.basename(img_filename))[0][:-5]
    candidate = os.path.join(mask_dir, f"{base}_mask.png")
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(mask_dir, f"{base}.png")
    if os.path.exists(candidate2):
        return candidate2
    return None


def mask_to_rgba(mask, alpha=0.45):
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    for lab in labels:
        rng = np.random.default_rng(int(lab) * 1234567)
        color = rng.random(3)
        m = (mask == lab)
        rgba[m, 0:3] = color
        rgba[m, 3] = alpha
    return rgba


class Viewer:
    def __init__(self, image_dir, mask_dir, alpha=0.45):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.alpha = float(alpha)

        self.images = list_images(image_dir)
        if not self.images:
            raise RuntimeError(f"No images found in: {image_dir}")

        self.idx = 0
        self.overlay_on = True

        # Cached current sample
        self.cur_img_fn = None
        self.cur_img = None
        self.cur_mask = None
        self.cur_overlay = None
        self.cur_mask_path = None

        # Matplotlib
        plt.rcParams['figure.autolayout'] = True
        self.fig = plt.figure()
        manager = self.fig.canvas.manager
        if hasattr(manager, 'window'):
            manager.window.state('zoomed')

        self.fig.canvas.mpl_connect('resize_event', self.on_figure_resize)

        self._setup_layout()

        # Create suptitle ONCE
        self.suptitle_text = self.fig.suptitle("", fontsize=14)

        # Artists (reuse!)
        self.im_artist = None
        self.ov_artist = None
        self.legend_artist = None

        # UI
        self._create_buttons()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current_sample()
        self.render_current_sample()
    
    def _setup_layout(self):
        self.fig.clear()
        
        # Responsive margins (as fraction of figure)
        left_margin = 0.05
        right_margin = 0.12
        bottom_margin = 0.05
        top_margin = 0.05
        
        # Calculate axes size
        ax_left = left_margin
        ax_width = 1 - left_margin - right_margin
        ax_bottom = bottom_margin
        ax_height = 1 - bottom_margin - top_margin
        
        self.ax = self.fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        self.ax.axis("off")
    
    def _create_buttons(self):
        """Create buttons with responsive positioning."""
        button_area_left = 0.82
        button_width = 0.15
        button_height_fraction = 0.06

        button_y_top = 0.85
        button_y_spacing = 0.12
        
        # Checkbox for overlay (top)
        ax_check = self.fig.add_axes([button_area_left, button_y_top - 0.08, button_width, 0.08])
        self.check = CheckButtons(ax_check, ["Overlay"], [self.overlay_on])
        self.check.on_clicked(self.on_toggle)
        
        # Navigation buttons (middle area)
        y_pos = button_y_top - button_y_spacing
        ax_back = self.fig.add_axes([button_area_left, y_pos, button_width, button_height_fraction])
        self.btn_back = Button(ax_back, "Back")
        self.btn_back.on_clicked(self.on_back)
        
        y_pos -= button_y_spacing
        ax_next = self.fig.add_axes([button_area_left, y_pos, button_width, button_height_fraction])
        self.btn_next = Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)
        
        y_pos -= button_y_spacing
        ax_del = self.fig.add_axes([button_area_left, y_pos, button_width, button_height_fraction])
        self.btn_del = Button(ax_del, "Delete")
        self.btn_del.on_clicked(self.on_delete)
    
    def on_figure_resize(self, event):
        if hasattr(self, 'ax'):
            self.render_current_sample()

    def _set_status(self, msg: str):
        self.suptitle_text.set_text(msg or "")
        self.fig.canvas.draw_idle()

    def load_current_sample(self):
        """Load image/mask/overlay only when index changes (or after delete)."""
        if not self.images:
            self.cur_img_fn = None
            self.cur_img = None
            self.cur_mask = None
            self.cur_overlay = None
            self.cur_mask_path = None
            return

        img_fn = self.images[self.idx]
        if img_fn == self.cur_img_fn:
            return  # already cached

        img_path = os.path.join(self.image_dir, img_fn)
        img = load_rgb(img_path)

        mask_path = build_mask_path(self.mask_dir, img_fn)
        mask = None
        overlay = None

        if mask_path is not None and os.path.exists(mask_path):
            mask = load_mask(mask_path)
            if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                mask = np.array(
                    Image.fromarray(mask).resize(
                        (img.shape[1], img.shape[0]),
                        resample=Image.NEAREST
                    )
                )
            overlay = mask_to_rgba(mask, alpha=self.alpha)

        self.cur_img_fn = img_fn
        self.cur_img = img
        self.cur_mask = mask
        self.cur_overlay = overlay
        self.cur_mask_path = mask_path

    def render_current_sample(self):
        """Render using reusable artists (fast + avoids buildup)."""
        if not self.images:
            self._set_status("Keine Bilder mehr vorhanden. Fenster wird geschlossen.")
            plt.close(self.fig)
            return

        img_fn = self.cur_img_fn
        title = f"{self.idx+1}/{len(self.images)}  |  {img_fn}"
        if self.cur_mask_path is None:
            title += "  |  (NO MASK FOUND)"

        fig_width_inch = self.fig.get_figwidth()
        dpi = self.fig.dpi
        fig_width_pixels = fig_width_inch * dpi
        title_fontsize = max(12, int(fig_width_pixels / 120))
        
        self.ax.set_title(title, fontsize=title_fontsize)

        # Image artist
        if self.im_artist is None:
            self.im_artist = self.ax.imshow(self.cur_img)
        else:
            self.im_artist.set_data(self.cur_img)

        # Overlay artist
        if self.cur_overlay is not None:
            if self.ov_artist is None:
                self.ov_artist = self.ax.imshow(self.cur_overlay)
            else:
                self.ov_artist.set_data(self.cur_overlay)
            self.ov_artist.set_visible(self.overlay_on)
        else:
            if self.ov_artist is not None:
                self.ov_artist.set_visible(False)

        # Legend artist
        if self.cur_mask is not None:
            labels = np.unique(self.cur_mask)
            labels = labels[labels != 0]
            if labels.size > 0:
                txt = "Labels: " + ", ".join(map(str, labels.tolist()))
                legend_fontsize = max(10, int(fig_width_pixels / 150))
                if self.legend_artist is None:
                    self.legend_artist = self.ax.text(
                        0.01, 0.01, txt,
                        transform=self.ax.transAxes,
                        fontsize=legend_fontsize,
                        va="bottom",
                        ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
                    )
                else:
                    self.legend_artist.set_text(txt)
                    self.legend_artist.set_fontsize(legend_fontsize)
                    self.legend_artist.set_visible(True)
            else:
                if self.legend_artist is not None:
                    self.legend_artist.set_visible(False)
        else:
            if self.legend_artist is not None:
                self.legend_artist.set_visible(False)

        self.fig.canvas.draw_idle()

    def on_next(self, _event=None):
        if not self.images:
            return
        self.idx = (self.idx + 1) % len(self.images)
        self._set_status("")
        self.load_current_sample()
        self.render_current_sample()

    def on_back(self, _event=None):
        if not self.images:
            return
        self.idx = (self.idx - 1) % len(self.images)
        self._set_status("")
        self.load_current_sample()
        self.render_current_sample()

    def on_toggle(self, _label=None):
        # IMPORTANT: do NOT reload/rebuild; just toggle visibility
        self.overlay_on = not self.overlay_on
        if self.ov_artist is not None:
            self.ov_artist.set_visible(self.overlay_on)
        self._set_status("")
        self.fig.canvas.draw_idle()

    def on_delete(self, _event=None):
        if not self.images:
            return

        img_fn = self.images[self.idx]
        img_path = os.path.join(self.image_dir, img_fn)
        mask_path = build_mask_path(self.mask_dir, img_fn)

        deleted = []
        errors = []

        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                deleted.append(f"image: {img_fn}")
            else:
                errors.append("image not found")
        except Exception as e:
            errors.append(f"image delete failed: {e}")

        if mask_path is not None:
            try:
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    deleted.append(f"mask: {os.path.basename(mask_path)}")
                else:
                    errors.append("mask not found")
            except Exception as e:
                errors.append(f"mask delete failed: {e}")

        # Update list/index/cache
        del self.images[self.idx]
        self.cur_img_fn = None  # force reload next render

        if self.images:
            if self.idx >= len(self.images):
                self.idx = len(self.images) - 1
            msg = ("Deleted: " + ", ".join(deleted)) if deleted else "Nothing deleted."
            if errors:
                msg += " | Errors: " + "; ".join(errors)
            self._set_status(msg)
            self.load_current_sample()
            self.render_current_sample()
        else:
            msg = "Deleted last item."
            if errors:
                msg += " Errors: " + "; ".join(errors)
            self._set_status(msg)
            plt.close(self.fig)

    def on_key(self, event):
        if event.key in ("n", "right"):
            self.on_next()
        elif event.key in ("b", "left"):
            self.on_back()
        elif event.key in ("o",):
            # keep checkbox state in sync
            self.check.set_active(0)  # triggers on_toggle
        elif event.key in ("d",):
            self.on_delete()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with original images")
    ap.add_argument("--masks", required=True, help="Folder with PNG masks (*_mask.png)")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay transparency (0..1)")
    args = ap.parse_args()

    Viewer(args.images, args.masks, alpha=args.alpha)
    plt.show()


if __name__ == "__main__":
    main()
