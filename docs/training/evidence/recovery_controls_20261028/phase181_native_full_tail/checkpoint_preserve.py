from pathlib import Path
import hashlib,json,shutil
p=Path('/home/genesis/agribot-data/recognition_campaign_20261024');out=p/'phase181_native_full_tail';ram=Path('/dev/shm/agribot-phase181-checkpoints')
assert (out/'result.json').exists()
files=[]
for filename in ('best.pt','last.pt'):
    source=out/filename;target=ram/filename
    if not source.is_symlink():
        assert not target.exists();source.rename(target);source.symlink_to(target)
    assert source.resolve()==target
    sha=hashlib.sha256(target.read_bytes()).hexdigest();files.append(dict(name=filename,bytes=target.stat().st_size,sha256=sha))
manifest=dict(files=files,source='Completed full-tail control, research only; safe incumbent and failed final checkpoint',volatile_on_gpu_host=True,laptop_backup_required=True)
(out/'checkpoint_backup.json').write_text(json.dumps(manifest,indent=2))
# Preserve all small run evidence across GPU-host reboot, without violating the
# persistent-disk reserve or discarding the RAM checkpoint originals.
durable=p/'phase181_initial_protocol_preflight'
for path in out.iterdir():
    if path.is_file() and path.suffix in ('.json','.npz'):shutil.copy2(path,durable/path.name)
print(json.dumps(manifest))
