"""Audit prospective tomato leaf membership without any model predictions.

The publisher combines field, internet and third-party material. Membership
surviving this screen is provisional, never proof of capture independence.
"""

from pathlib import Path, PurePosixPath
import argparse
import collections
import concurrent.futures
import hashlib
import json
import zipfile
import numpy as np

from campaign_data import profile
from capture_families import geometric_edges

MAPPING = {"Tomato healthy leaf": "Healthy", "Tomato septoria leaf spot": "Other",
           "Tomato verticulium wilt": "Other"}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            value.update(chunk)
    return value.hexdigest()


def immutable(path,value):
    text=json.dumps(value,indent=2)+"\n"
    if path.exists():
        assert path.read_text()==text, ("Sealed identity changed",str(path))
    else:
        temporary=path.with_suffix(".tmp")
        temporary.write_text(text); temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign); out=p/"phase86_tcp_source"
    audit=json.loads((out/"download_audit.json").read_text())
    assert not audit["diagnostic_predictions_opened"] and not audit["training_used"]
    archive=out/"publisher_archive.zip"
    assert digest(archive)==audit["verified_sha256"]
    target=(out/"files").resolve(); target.mkdir(exist_ok=True)
    rows=[]
    with zipfile.ZipFile(archive) as zipped:
        for member in sorted(zipped.infolist(),key=lambda item:item.filename):
            name=PurePosixPath(member.filename)
            if member.is_dir() or name.parent.name not in MAPPING or name.suffix.lower() not in (".jpg",".jpeg",".png",".bmp"):
                continue
            assert not name.is_absolute() and ".." not in name.parts
            path=target.joinpath(*name.parts).resolve()
            assert path.is_relative_to(target)
            assert (member.external_attr>>16)&0o170000 != 0o120000, "Archive symlink refused"
            path.parent.mkdir(parents=True,exist_ok=True)
            content=zipped.read(member)
            if path.exists():
                assert digest(path)==hashlib.sha256(content).hexdigest()
            else:
                path.write_bytes(content)
            rows.append(dict(id="tcp4:"+str(name),path=str(path),label=MAPPING[name.parent.name],
                             raw_label=name.parent.name,source="tcp_mixed_publisher_v4",species="tomato",
                             box=None,split="provisional_sealed_external",licence="CC BY 4.0",
                             source_url=audit["source_url"],source_version=audit["source_doi"],
                             label_scope="Publisher image-level tomato leaf condition; septoria and verticillium are unsupported Other, not named supported diseases"))
    assert len(rows)==3668
    profile_path=out/"quality_profiles.json"
    if profile_path.exists():
        existing=json.loads(profile_path.read_text())
        assert len(existing)==len(rows)
        for row,saved in zip(rows,existing):
            assert row.items()<=saved.items() and digest(row["path"])==saved["file_sha256"]
        rows=existing
    else:
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for row,(metadata,_) in zip(rows,pool.map(profile,rows)):
                row.update(metadata)
        immutable(profile_path,rows)
    print(json.dumps(dict(stage="profiled",images=len(rows))),flush=True)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        parent[find(j)]=find(i)
    duplicates=[]
    for i,row in enumerate(rows):
        for j in range(i):
            other=rows[j]
            if any(row[k]==other[k] for k in ("file_sha256","rgb32_sha256","model_input_sha256")) or (int(row["phash"],16)^int(other["phash"],16)).bit_count()<=2:
                union(i,j); duplicates.append([i,j])
    duplicate_groups=collections.defaultdict(list)
    for i in range(len(rows)):
        duplicate_groups[find(i)].append(i)
    excluded={}
    for ids in duplicate_groups.values():
        if len({rows[i]["label"] for i in ids})>1:
            for i in ids: excluded[i]="Contradictory exact/near-copy labels"
        else:
            for i in ids[1:]: excluded[i]="Repeated image, one representative retained"
    geometry_path=out/"geometry.json"
    if geometry_path.exists():
        geometry=json.loads(geometry_path.read_text())
        assert geometry["profiles_sha256"]==digest(profile_path)
    else:
        edges,candidates=geometric_edges(rows)
        geometry=dict(profiles_sha256=digest(profile_path),edges=edges,candidates=candidates)
        immutable(geometry_path,geometry)
    for edge in geometry["edges"]:
        union(edge["first"],edge["second"])
    components=collections.defaultdict(list)
    for i in range(len(rows)): components[find(i)].append(i)
    for ids in components.values():
        group="tcp4:inferred_family:"+hashlib.sha256("|".join(rows[i]["id"] for i in ids).encode()).hexdigest()[:20]
        for i in ids: rows[i]["group"]=group
    old=[]; prior={}
    manifests=["original_quality_profile.json","new_development_manifest.json","locked_new_source_manifest.json",
               "locked_binary_manifest.json","mixed_stress_manifest.json","phase2_source_manifest.json",
               "phase6_roi/cnn_inputs/rows.json","phase6_roi/reserved_manifest.json",
               "phase13_siglip/admission_manifest.json","phase9_ad_source/sealed_geometric_manifest.json",
               "phase24_olid_source/sealed_manifest.json","phase84_population_head/fitting_rows.json"]
    for relative in manifests:
        file=p/relative
        if file.exists():
            prior[relative]=digest(file); old.extend(json.loads(file.read_text()))
    old_files={r["file_sha256"] for r in old if r.get("file_sha256")}
    # Include every source file behind the original ROI features, not just a split.
    source_files=p/"phase29_compact/source_files.json"
    prior[str(source_files.relative_to(p))]=digest(source_files)
    old_files.update(json.loads(source_files.read_text()).values())
    old_rgb={r["rgb32_sha256"] for r in old if r.get("rgb32_sha256")}
    old_input={r["model_input_sha256"] for r in old if r.get("model_input_sha256")}
    for relative in ["phase29_compact/inputs.json","phase38_context_spatial/new_inputs.json","phase46_paper_primary/new_inputs.json","phase87_diagnostic_bag_features/new_inputs.json"]:
        file=p/relative
        if file.exists():
            prior[relative]=digest(file)
            old_input.update(r["input_sha256"] for r in json.loads(file.read_text()))
    old_phash=np.array(sorted({int(r["phash"],16) for r in old if r.get("phash")}),np.uint64)
    assert len(old_files)>30000 and len(old_phash)>10000
    overlapping=set()
    for i,row in enumerate(rows):
        distance=int(np.bitwise_count(old_phash ^ np.uint64(int(row["phash"],16))).min())
        row["nearest_prior_phash_distance"]=distance
        # Conservative eight-bit exclusion also quarantines near-public copies.
        if row["file_sha256"] in old_files or row["rgb32_sha256"] in old_rgb or row["model_input_sha256"] in old_input or distance<=8:
            overlapping.add(row["group"])
        if min(row["width"],row["height"])<128:
            excluded[i]="Insufficient native resolution for leaf-condition adjudication"
    for i,row in enumerate(rows):
        if row["group"] in overlapping:
            excluded[i]="Entire inferred family overlaps or closely resembles consumed/training material"
    accepted=[r for i,r in enumerate(rows) if i not in excluded]
    immutable(out/"provisional_sealed_manifest.json",accepted)
    healthy=[r for r in accepted if r["label"]=="Healthy"]
    disease=[r for r in accepted if r["label"]!="Healthy"]
    counts_ok=len(healthy)>=100 and len(disease)>=300 and len({r["group"] for r in healthy})>=30 and len({r["group"] for r in disease})>=60
    report=dict(raw_n=len(rows),provisional_n=len(accepted),healthy_n=len(healthy),nonhealthy_n=len(disease),
                healthy_families=len({r["group"] for r in healthy}),nonhealthy_families=len({r["group"] for r in disease}),
                counts_sufficient=counts_ok,independence_established=False,acceptance_ready=False,
                reason="Mixed publisher provenance requires visual/source adjudication; overlap screening alone cannot establish fresh capture independence",
                prior_manifests=prior,prior_file_identities=len(old_files),prior_phashes=len(old_phash),
                exclusions=[dict(id=rows[i]["id"],reason=reason) for i,reason in excluded.items()],
                source_sha256=digest(__file__),profiles_sha256=digest(profile_path),geometry_sha256=digest(geometry_path),
                provisional_manifest_sha256=digest(out/"provisional_sealed_manifest.json"),
                diagnostic_predictions_opened=False,training_used=False,
                limitations="Inferred geometric families are not verified plant IDs. No exhaustive cross-source SIFT search. Source has only unsupported-disease classes; it cannot replace all supported-disease recognition gates.")
    immutable(out/"membership_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("prior_manifests","exclusions")}),flush=True)


if __name__=="__main__":
    main()
