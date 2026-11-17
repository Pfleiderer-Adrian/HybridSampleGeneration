# Postprocessing materialized hybrids

Postprocessing is an optional quality check that runs after materialization.
It evaluates every hybrid against configurable rules and records an
accept/reject decision. Generated images and masks are never modified or
deleted.

```python
config.postprocessing.enabled = True
generator.materialize_hybrid_samples()
summary = generator.postprocess()
```

A call with `config.postprocessing.enabled = False` is a no-op.

## Rules

Each rule is evaluated on the union of the final materialized anomaly masks of
a hybrid. Any failed rule marks the entire hybrid as failed.

- `reject_empty_mask`: the hybrid contains no anomaly pixels.
- `mask_area_fraction`: the anomaly area, as a fraction of the image, lies
  outside `[minimum, maximum]`.
- `minimum_object_overlap`: less than `threshold` of the anomaly pixels lie on
  the imaged object. This detects anomalies placed on the background.

## Object mask

`minimum_object_overlap` needs an object mask. It is estimated from the
unmodified healthy original of each hybrid using a fixed intensity threshold in
original image units (0–255):

1. Reduce channels: per-pixel channel minimum (`minimum`) or RGB luminance
   (`grayscale`).
2. Keep pixels `below` or `above` `threshold`.
3. Keep the `largest` connected component or `all` components of at least
   `min_component_area` pixels/voxels.
4. Optionally fill enclosed holes (`fill_holes`).

Choose the settings to match your data: for example, a dark object on a bright
background uses `comparison = "below"`, a bright object on a dark background
uses `comparison = "above"`. Objects consisting of several separate parts need
`components = "all"`.

```python
post = config.postprocessing
post.enabled = True
post.action = "report"
post.save_previews = True
post.foreground.channel_reduction = "minimum"
post.foreground.comparison = "below"
post.foreground.threshold = 240
post.foreground.components = "largest"
post.foreground.fill_holes = True
post.rules.minimum_object_overlap.enabled = True
post.rules.minimum_object_overlap.threshold = 1.0
```

Threshold masks are heuristics, not semantic ground truth. Dark object
structures, low contrast and anomalies that legitimately change the object's
outline can fail the overlap rule. Start with `action = "report"` and
`save_previews = True`, inspect the results, and only then switch to
`exclude`. An empty estimated object mask stops evaluation instead of silently
rejecting every hybrid. The overlap rule requires healthy originals as
placement targets.

## Actions and outputs

- `report`: record decisions only.
- `exclude`: additionally signals downstream consumers to omit failed hybrids.

Each run writes `postprocessing/<timestamp_id>/` in the study folder with
`configuration.json`, `summary.json`, `decisions.csv` and, if enabled,
2D previews (object mask in blue, anomalies in red). Reports of earlier runs
are kept, so changed settings can be compared. The current decisions are
stored in the `postprocessing_decisions` table of the study database; the
materialization status of the hybrids remains unchanged. The visualizer shows
the saved decision and reasons in the hybrid details.

## Using decisions downstream

The generator itself does not filter anything. Code that consumes the hybrids,
such as a training data loader, reads the decision per hybrid:

```python
from hybrid_sample_generator.postprocessing.service import decision_for

decision = decision_for(repository, artifact_store, hybrid, original, config.postprocessing)
if config.postprocessing.action == "exclude" and not decision["accepted"]:
    ...  # skip this hybrid
```

`decision_for` returns `None` when postprocessing is disabled. When it is
enabled, it raises an error if the decision is missing or stale, so a consumer
never uses outdated results.

## Staleness

A decision is stale when either of the following has changed since it was
computed:

- the postprocessing configuration (compared via a configuration fingerprint);
- the original image, hybrid image or hybrid mask (compared via path, file size
  and nanosecond modification time; these are not content hashes).

Rematerialization clears all current decisions before fusion starts, and
replacing a plan deletes the decisions of its hybrids. In both cases, run
`postprocess()` again.

## Limitations

- Rules apply to the whole hybrid, not to individual placements.
- Defect types are not classified; type-specific rules are not supported.
- The object mask is threshold-based only; learned object masks are not
  supported.
- Previews are written for 2D images only; the checks themselves support 2D
  and 3D, grayscale and RGB.
