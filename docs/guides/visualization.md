# Visualization

`run_hybrid_visualizer(config)` opens a repository-backed study browser with
six views: study overview, datasource originals, real/synthetic anomaly variants,
hybrid samples and their placements, metric-based evaluation, and the complete
normalized data structure. The Datasource tab lists all ingested originals with
source-name/ID search and filters for anomalous/control and annotated/unannotated
samples. Images use automatic RGB display for three-channel arrays; channel,
slice, contrast and mask overlays remain selectable for grayscale and 3D data.
In every image view, use the mouse wheel to zoom around the pointer and drag with
the left mouse button to pan. Double-click a panel or use **Reset zoom** to fit
images again. Zoom persists across contrast, channel, mask and slice changes;
selecting another sample resets it. Use Shift+wheel, the slice slider, or Up/Down
keys to navigate 3D slices.
The Evaluation tab also previews linked fused placement ROIs for cutout metrics.
Use **Placement ROI preview** to choose among multiple placements of the same
synthetic anomaly; available ROI files are preferred initially. Placement metrics
always show their evaluated placement, and selecting a preview leaves the metrics
and evaluation scope unchanged.
Artifacts are loaded lazily and cached only while they are inspected. The data
structure view can preview dependent records before moving their files into a
recoverable `.trash` folder and removing the corresponding database records.

The visualizer can also be started for an existing study folder:

```bash
python -m hybrid_sample_generator.visualization /path/to/study --channel auto
```
