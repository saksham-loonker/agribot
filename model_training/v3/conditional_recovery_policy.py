"""A new admission expert cannot undo existing Healthy or named-disease results."""
def recover_only(row,retained,packet,probability,manifest):
    if retained['app_replay']['kind'] in ('HEALTHY','DISEASE'):
        return retained
    from phase88_diagnostic_bag_head import route_paired
    # Existing whole/genus/compact evidence and suspicion guards still apply.
    # This expert learns admission only; it adds no new veto to existing Healthy.
    return route_paired(row,retained,packet,probability,1.,.95,manifest)
