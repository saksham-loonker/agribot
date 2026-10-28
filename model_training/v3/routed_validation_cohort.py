"""Keep full-cohort Healthy denominators when a head fits only routed cases."""
def healthy_cohort(rows,active_paths,retained,source):
    cohort=[row for row in rows if row['validation'] and row['source']==source and row['y']==1]
    paths=[row['path'] for row in cohort]
    if len(paths)!=len(set(paths)):raise ValueError('Validation cohort contains duplicate photographs')
    if any(path not in retained for path in paths):raise ValueError('Missing retained cohort verdict')
    fixed=sum(retained[path]['app_replay']['kind']=='HEALTHY' for path in paths if path not in active_paths)
    return len(paths),fixed
