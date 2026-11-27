# Visualization

The study browser is a desktop application for inspecting a study at any
pipeline stage. It reads records from the study database and loads arrays only
while they are inspected.

## Starting the browser

From Python, for the study of a configuration:

```python
from hybrid_sample_generator.visualization import run_hybrid_visualizer

config.save_config_file()
run_hybrid_visualizer(config)
```

From the command line, for any study folder:

```bash
python -m hybrid_sample_generator.visualization /path/to/study --channel auto
```

The call blocks until the window is closed. The browser needs a graphical
session and Tk; see [installation](../getting-started/installation.md). Save
the configuration beforehand: the Overview tab shows the saved
`configuration.json`.

`--channel` (or the `channel` argument) selects what is displayed: `auto` shows
three-channel arrays as RGB, a number shows that single channel.

## Tabs

| Tab | Content |
| --- | --- |
| **Overview** | record counts, study health checks such as missing artifact references, and the saved configuration |
| **Datasource** | all ingested originals with filters for anomalous/control and annotated/unannotated samples |
| **Anomalies** | original → real anomaly → synthetic variants, with cutouts, masks and ROIs |
| **Hybrids & placements** | original → hybrid → placement, with a status filter (`generated`, `planned`, `failed`), an optional absolute-difference view and the saved [postprocessing](postprocessing.md) decision |
| **Evaluation** | [evaluation](evaluation.md) metric differences and outliers per real/synthetic pair, restricted to cutout or placement comparisons with the *Scope* filter |
| **Data structure** | the complete record hierarchy, including removal of records |

The search field in the header filters by record ID or source name.
**Refresh** reloads the study after a pipeline stage has changed it. The footer
shows the IDs of the current selection.

## Image controls

| Action | Control |
| --- | --- |
| Zoom around the pointer | mouse wheel |
| Pan | drag with the left mouse button |
| Fit the images again | double-click a panel or **Reset zoom** |
| Previous / next record | Left / Right keys or the buttons below the list |
| Change the 3D slice | Shift + wheel, the slice slider, or Up / Down keys |
| Mask overlays, opacity, channel, contrast | controls in the side panel |

Zoom persists across contrast, channel, mask and slice changes; selecting
another sample resets it.

## Evaluation tab

The Evaluation tab lists the pairs of the last
[`evaluate_study()`](evaluation.md) run and marks outliers; the slider sets
which top share of the differences counts as outlier in this view.

For cutout metrics, the tab also previews the fused placement ROIs that use
the selected synthetic anomaly. Use **Placement ROI preview** to choose among
several placements of the same synthetic anomaly; placements with stored ROI
files are preferred initially. Placement metrics always show their evaluated
placement, and selecting a preview leaves the metrics and the evaluation scope
unchanged.

## Removing records

In the Data structure tab, **Archive and remove selected entity…** first
previews every dependent record: removing a real anomaly, for example, also
removes its synthetic variants and their placements. After confirmation, the
files are moved into a `.trash` folder inside the study folder, from where they
can be restored manually, and the database records are deleted.
**Delete selected hybrid sample…** in the Hybrids tab does the same for one
hybrid.
