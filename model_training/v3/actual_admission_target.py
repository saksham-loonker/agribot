"""Training actionability must match the final retained app route exactly."""
from conditional_recovery_policy import recover_only


def actual_admission_target(original,retained,packet,manifest):
    # Clinical labels do not enter eligibility. Protected old decisions remain
    # protected; a probability of one tests the exact unchanged final guards.
    if retained['app_replay']['kind'] in ('HEALTHY','DISEASE'):return False
    return recover_only(original,retained,packet,1.,manifest)['app_replay']['kind']=='HEALTHY'
