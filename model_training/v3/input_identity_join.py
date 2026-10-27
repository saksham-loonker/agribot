"""Join independently numbered input caches by canonical input SHA, never index."""


def input_identity_join(target_rows, donor_rows, donor_tables):
    lookup={}
    for offset,table in enumerate(donor_tables):
        key=donor_rows[int(table)]["input_sha256"]
        if not isinstance(key,str) or len(key)!=64:
            raise ValueError("Canonical input SHA-256 required")
        lookup.setdefault(key,offset)
    joined=[]
    for row in target_rows:
        key=row["input_sha256"]
        if not isinstance(key,str) or len(key)!=64:
            raise ValueError("Canonical input SHA-256 required")
        joined.append(lookup.get(key))
    return joined
