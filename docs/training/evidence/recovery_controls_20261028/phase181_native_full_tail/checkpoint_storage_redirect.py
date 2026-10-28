from pathlib import Path
import shutil,json
p=Path('/home/genesis/agribot-data/recognition_campaign_20261024')
out=p/'phase181_native_full_tail';ram=Path('/dev/shm/agribot-phase181-output')
assert not out.is_symlink() and not ram.exists()
assert not (out/'warm_parity.json').exists(),'Redirect before checkpoint writes'
assert shutil.disk_usage('/dev/shm').free>(800<<20)
ram.mkdir()
for path in out.iterdir():
    assert path.is_symlink() or path.name=='protocol.json',path
    if path.is_symlink():(ram/path.name).symlink_to(path.readlink())
    else:shutil.copy2(path,ram/path.name)
backup=p/'phase181_initial_protocol_preflight';assert not backup.exists()
out.rename(backup);out.symlink_to(ram,target_is_directory=True)
assert (out/'protocol.json').read_bytes()==(backup/'protocol.json').read_bytes()
print(json.dumps(dict(output=str(out.resolve()),checkpoint_writes='Atomic temporary and final writes both remain on RAM; persistent disk reserve preserved',training_interrupted=False)))
