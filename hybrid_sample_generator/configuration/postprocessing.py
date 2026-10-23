"""Optional, non-destructive quality checks for materialized hybrids."""
from dataclasses import dataclass, field, asdict
import math

@dataclass
class ForegroundConfiguration:
    method: str = "threshold"
    channel_reduction: str = "minimum"
    comparison: str = "below"
    threshold: float = 240.0
    components: str = "largest"
    min_component_area: int = 1
    fill_holes: bool = True

@dataclass
class OverlapRule:
    enabled: bool = False
    threshold: float = 1.0
    exempt_source_groups: list[str] = field(default_factory=list)

@dataclass
class AreaRule:
    enabled: bool = False
    minimum: float = 0.0
    maximum: float = 1.0

@dataclass
class RulesConfiguration:
    minimum_object_overlap: OverlapRule = field(default_factory=OverlapRule)
    mask_area_fraction: AreaRule = field(default_factory=AreaRule)
    reject_empty_mask: bool = True

@dataclass
class PostprocessingConfiguration:
    enabled: bool = False
    action: str = "report"
    save_previews: bool = False
    foreground: ForegroundConfiguration = field(default_factory=ForegroundConfiguration)
    rules: RulesConfiguration = field(default_factory=RulesConfiguration)

    def validate(self):
        flags=[self.enabled,self.save_previews,self.foreground.fill_holes,
               self.rules.reject_empty_mask,self.rules.minimum_object_overlap.enabled,
               self.rules.mask_area_fraction.enabled]
        if any(type(v) is not bool for v in flags):
            raise ValueError("Postprocessing enabled flags must be boolean.")
        if self.action not in {"report", "exclude"}:
            raise ValueError("postprocessing.action must be report or exclude.")
        f=self.foreground
        if (f.method!="threshold" or f.channel_reduction not in {"minimum","grayscale"}
                or f.comparison not in {"below","above"} or f.components not in {"largest","all"}):
            raise ValueError("Unsupported postprocessing foreground settings.")
        if not math.isfinite(f.threshold) or not 0<=f.threshold<=255:
            raise ValueError("Foreground threshold must be finite and in [0,255].")
        if type(f.min_component_area) is not int or f.min_component_area<1:
            raise ValueError("min_component_area must be a positive pixel/voxel count.")
        groups=self.rules.minimum_object_overlap.exempt_source_groups
        if not isinstance(groups,list) or any(not isinstance(g,str) or not g for g in groups) or len(set(groups))!=len(groups):
            raise ValueError("exempt_source_groups must be distinct nonempty source-group names.")
        a=self.rules.mask_area_fraction;o=self.rules.minimum_object_overlap
        if any(not math.isfinite(v) or not 0<=v<=1 for v in [a.minimum,a.maximum,o.threshold]) or a.minimum>a.maximum:
            raise ValueError("Postprocessing fraction thresholds must satisfy 0 <= minimum <= maximum <= 1.")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls,values):
        values=dict(values);f=ForegroundConfiguration(**values.pop('foreground',{}))
        r=dict(values.pop('rules',{}));o=OverlapRule(**r.pop('minimum_object_overlap',{}));a=AreaRule(**r.pop('mask_area_fraction',{}))
        obj=cls(**values,foreground=f,rules=RulesConfiguration(**r,minimum_object_overlap=o,mask_area_fraction=a))
        obj.validate();return obj
