#!/usr/bin/env python3
"""
Mask Visualizer (Image + Segmentation Overlay)

Inputs:
  --images  Path to folder containing the original images
  --masks   Path to folder containing the exported PNG masks (*_mask.png)
  --alpha   Overlay transparency (0..1), default 0.45

Features:
  - Shows one sample at a time
  - Toggle overlay ON/OFF
  - "Next" button to show next image
  - Keyboard shortcuts:
      n  -> next
      o  -> toggle overlay
      q  -> quit

Example:
  python visualize_masks.py --images /path/to/images --masks /path/to/masks_png
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(IMG_EXTS):
            files.append(fn)
    files.sort()
    return files


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    # Keep integer labels (L or I;16)
    return np.array(Image.open(path))


def build_mask_path(mask_dir, img_filename):
    base = os.path.splitext(os.path.basename(img_filename))[0]
    # Default naming from exporter: <base>_mask.png
    candidate = os.path.join(mask_dir, f"{base}_mask.png")
    if os.path.exists(candidate):
        return candidate

    # Fallback: allow same stem without _mask
    candidate2 = os.path.join(mask_dir, f"{base}.png")
    if os.path.exists(candidate2):
        return candidate2

    return None


def mask_to_rgba(mask, alpha=0.45):
    """
    Convert label mask into an RGBA overlay image.
    Background (0) is fully transparent.
    Each class id gets a stable pseudo-random color.
    """
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    labels = np.unique(mask)
    labels = labels[labels != 0]  # skip background

    for lab in labels:
        # Stable color derived from label id (no fixed palette needed)
        rng = np.random.default_rng(int(lab) * 1234567)
        color = rng.random(3)  # RGB in 0..1

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

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_axes([0.05, 0.10, 0.80, 0.85])  # main
        self.ax.axis("off")

        # UI: Next button
        ax_next = self.fig.add_axes([0.88, 0.22, 0.10, 0.08])
        self.btn_next = Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)

        # UI: checkbox overlay
        ax_check = self.fig.add_axes([0.88, 0.35, 0.10, 0.10])
        self.check = CheckButtons(ax_check, ["Overlay"], [self.overlay_on])
        self.check.on_clicked(self.on_toggle)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw_current()

    def draw_current(self):
        img_fn = self.images[self.idx]
        img_path = os.path.join(self.image_dir, img_fn)
        img = load_rgb(img_path)

        mask_path = build_mask_path(self.mask_dir, img_fn)
        mask = None
        overlay = None

        if mask_path is not None and os.path.exists(mask_path):
            mask = load_mask(mask_path)

            # Ensure mask matches image size
            if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                mask = np.array(
                    Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
                )

            overlay = mask_to_rgba(mask, alpha=self.alpha)

        self.ax.clear()
        self.ax.axis("off")

        title = f"{self.idx+1}/{len(self.images)}  |  {img_fn}"
        if mask_path is None:
            title += "  |  (NO MASK FOUND)"
        self.ax.set_title(title, fontsize=12)

        self.ax.imshow(img)

        if self.overlay_on and overlay is not None:
            self.ax.imshow(overlay)

        # Optional: show present labels
        if mask is not None:
            labels = np.unique(mask)
            labels = labels[labels != 0]
            if labels.size > 0:
                legend_text = "Labels: " + ", ".join(map(str, labels.tolist()))
                self.ax.text(
                    0.01, 0.01, legend_text,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    va="bottom",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
                )

        self.fig.canvas.draw_idle()

    def on_next(self, _event=None):
        self.idx = (self.idx + 1) % len(self.images)
        self.draw_current()

    def on_toggle(self, _label=None):
        self.overlay_on = not self.overlay_on
        self.draw_current()

    def on_key(self, event):
        if event.key in ("n", "right"):
            self.on_next()
        elif event.key in ("o",):
            # toggle overlay & checkbox state
            self.check.set_active(0)  # triggers on_toggle
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
