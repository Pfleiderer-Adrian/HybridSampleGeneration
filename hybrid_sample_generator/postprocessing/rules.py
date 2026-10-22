"""Explicit reasons for rejecting a materialized hybrid."""
def violations(values,rules):
    reasons=[]
    if rules.reject_empty_mask and not values['mask_pixels']:reasons.append('empty_mask')
    area=rules.mask_area_fraction
    if area.enabled and not area.minimum<=values['mask_area_fraction']<=area.maximum:
        reasons.append('mask_area_fraction')
    overlap=rules.minimum_object_overlap
    if overlap.enabled and values['object_overlap']<overlap.threshold:
        reasons.append('minimum_object_overlap')
    return reasons
