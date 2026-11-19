"""Optional, non-destructive quality checks for materialized hybrids."""
from dataclasses import asdict, dataclass

from hybrid_sample_generator.configuration.parameters import param, validate_parameters


@dataclass
class ForegroundConfiguration:
    method: str = param(
        "threshold", choices=("threshold",), doc="Object-mask estimator.",
    )
    channel_reduction: str = param(
        "minimum",
        choices=("minimum", "grayscale"),
        doc="Minimum channel value (`minimum`) or RGB luminance (`grayscale`).",
    )
    comparison: str = param(
        "below", choices=("below", "above"), doc="Threshold direction, strict comparison.",
    )
    threshold: float = param(
        240.0,
        ge=0.0,
        le=255.0,
        doc="Threshold in original image units; no per-image normalization.",
    )
    components: str = param(
        "largest",
        choices=("largest", "all"),
        doc="Keep the largest component (`largest`) or `all`.",
    )
    min_component_area: int = param(
        1,
        ge=1,
        doc="Minimum connected-component pixel/voxel count, before hole filling.",
    )
    fill_holes: bool = param(True, doc="Fill enclosed holes after component selection.")


@dataclass
class OverlapRule:
    enabled: bool = param(
        False, doc="Measure anomaly coverage by the estimated object mask.",
    )
    threshold: float = param(
        1.0,
        ge=0.0,
        le=1.0,
        doc="Minimum fraction of anomaly pixels inside the object; inclusive bound.",
    )
    exempt_source_names: list[str] = param(
        default_factory=list,
        doc="Exact `source_name`s of donor originals whose placements are not overlap-checked; set before materialization. Unknown names produce a warning.",
    )


@dataclass
class AreaRule:
    enabled: bool = param(False, doc="Enable whole-image anomaly area bounds.")
    minimum: float = param(0.0, ge=0.0, le=1.0, doc="Inclusive lower bound.")
    maximum: float = param(1.0, ge=0.0, le=1.0, doc="Inclusive upper bound.")


@dataclass
class RulesConfiguration:
    minimum_object_overlap: OverlapRule = param(default_factory=OverlapRule)
    mask_area_fraction: AreaRule = param(default_factory=AreaRule)
    reject_empty_mask: bool = param(
        True, doc="Mark empty materialized anomaly masks as failed.",
    )


@dataclass
class PostprocessingConfiguration:
    enabled: bool = param(
        False, doc="Require and run quality checks when explicitly called.",
    )
    action: str = param(
        "report",
        choices=("report", "exclude"),
        doc="`report` records decisions; `exclude` additionally signals downstream consumers to omit failed hybrids. Neither deletes artifacts.",
    )
    save_previews: bool = param(
        False,
        doc="Write 2D original overlays with foreground in blue and anomalies in red.",
    )
    foreground: ForegroundConfiguration = param(default_factory=ForegroundConfiguration)
    rules: RulesConfiguration = param(default_factory=RulesConfiguration)

    def validate(self):
        validate_parameters(self, "postprocessing")
        names = self.rules.minimum_object_overlap.exempt_source_names
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                "postprocessing.rules.minimum_object_overlap.exempt_source_names "
                "must be distinct nonempty original source names."
            )
        area = self.rules.mask_area_fraction
        if area.minimum > area.maximum:
            raise ValueError(
                "postprocessing.rules.mask_area_fraction.minimum must not exceed maximum."
            )

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls,values):
        values=dict(values);f=ForegroundConfiguration(**values.pop('foreground',{}))
        r=dict(values.pop('rules',{}));o=OverlapRule(**r.pop('minimum_object_overlap',{}));a=AreaRule(**r.pop('mask_area_fraction',{}))
        obj=cls(**values,foreground=f,rules=RulesConfiguration(**r,minimum_object_overlap=o,mask_area_fraction=a))
        obj.validate();return obj
