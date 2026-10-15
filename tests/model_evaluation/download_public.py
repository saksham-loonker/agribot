"""Download pinned public, labeled diagnostic samples; never scrape search thumbnails."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, io, json, random, time
from collections import Counter
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dist/model-evaluation-20261015'
SOURCE = OUT / 'source'
SOURCE.mkdir(parents=True, exist_ok=True)
SEED = 20261015


def fetch(url, destination):
    if destination.exists():
        return destination.read_bytes()
    last = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 90), headers={'User-Agent': 'Agribot-public-model-diagnostic'})
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise last


def tree(repo, branch):
    if repo in {'mamta-joshi-gehlot/Tomato-Village', 'spMohanty/PlantVillage-Dataset'}:
        name = repo.split('/')[1]
        root = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{branch}', SOURCE / (name + '_root.json')))
        revision = root['sha']
        entries = root['tree']
        prefix = ''
        segments = ['Variant-a(Multiclass Classification)'] if 'Tomato-Village' in repo else ['raw', 'color']
        for depth, segment in enumerate(segments):
            sha = next(e['sha'] for e in entries if e['path'] == segment)
            recursive = '?recursive=1' if depth == len(segments)-1 else ''
            subtree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{sha}{recursive}', SOURCE / (name + f'_subtree{depth}.json')))
            if subtree.get('truncated'):
                if repo != 'spMohanty/PlantVillage-Dataset' or segment != 'color':
                    raise RuntimeError('Truncated scoped dataset index')
                flat = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{sha}', SOURCE / 'pv_color_dirs.json'))
                combined = []
                for category in flat['tree']:
                    if category['type'] != 'tree' or not category['path'].startswith('Tomato___'): continue
                    cat = category['path']
                    sub = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{category["sha"]}?recursive=1', SOURCE / (cat + '_index.json')))
                    if sub.get('truncated'): raise RuntimeError('Truncated category: ' + cat)
                    combined.extend({**e, 'path': cat + '/' + e['path']} for e in sub['tree'])
                subtree = {'tree': combined}
            entries = subtree['tree']; prefix += segment + '/'
        return revision, [e for e in root['tree'] if e['type'] == 'blob'] + [{**e, 'path': prefix + e['path']} for e in entries]
    data = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1', SOURCE / (repo.split('/')[1] + '_tree.json')))
    if data.get('truncated'):
        if repo != 'spMohanty/PlantVillage-Dataset':
            raise RuntimeError(f'Truncated source index: {repo}')
        root = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{data["sha"]}', SOURCE / 'pv_root.json'))
        raw_sha = next(e['sha'] for e in root['tree'] if e['path'] == 'raw')
        raw_tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{raw_sha}', SOURCE / 'pv_raw.json'))
        color_sha = next(e['sha'] for e in raw_tree['tree'] if e['path'] == 'color')
        color_tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{color_sha}?recursive=1', SOURCE / 'pv_color.json'))
        if color_tree.get('truncated'): raise RuntimeError('Truncated PlantVillage color index')
        data['tree'] = [{**e, 'path': 'raw/color/' + e['path']} for e in color_tree['tree']]
    return data['sha'], data['tree']


def raw(repo, revision, path):
    return f'https://raw.githubusercontent.com/{repo}/{revision}/' + quote(path, safe='/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-only', action='store_true')
    args = parser.parse_args()
    repos = [('tomatovillage', 'mamta-joshi-gehlot/Tomato-Village', 'main'),
             ('plantdoc', 'pratikkayal/PlantDoc-Object-Detection-Dataset', 'master'),
             ('plantvillage', 'spMohanty/PlantVillage-Dataset', 'master')]
    indices = {}
    for source, repo, branch in repos:
        try:
            rev, entries = tree(repo, branch)
        except requests.HTTPError as exc:
            if exc.response.status_code != 404: raise
            rev, entries = tree(repo, 'master' if branch == 'main' else 'main')
        indices[source] = (repo, rev, entries)
        print(source, rev, 'files:', len(entries), flush=True)
        for suffix in ['README.md', 'LICENSE', 'LICENSE.txt', 'LICENSE.md']:
            if any(e['path'] == suffix for e in entries):
                fetch(raw(repo, rev, suffix), SOURCE / (source + '_' + suffix))
    (SOURCE / 'revisions.json').write_text(json.dumps({s: {'repo': v[0], 'commit': v[1]} for s, v in indices.items()}, indent=2))
    if args.index_only:
        for source, (_, _, entries) in indices.items():
            print(source, 'top paths:', [e['path'] for e in entries[:12]], flush=True)
        return
    jobs, records = [], []
    def add(source, path, label, split, objects=None):
        repo, revision, _ = indices[source]
        local = OUT / 'images' / source / (hashlib.sha256(path.encode()).hexdigest()[:16] + Path(path).suffix.lower())
        record = dict(id=source + '_' + local.stem, source=source, source_path=path,
                      source_url=raw(repo, revision, path), source_commit=revision,
                      label=label, split=split, path=str(local.relative_to(ROOT)), objects=objects or [])
        jobs.append((record, local)); records.append(record)
    # Exhaust the authors' original Variant-A test split, all eight supported classes.
    for e in indices['tomatovillage'][2]:
        parts = e['path'].split('/')
        if len(parts) >= 4 and parts[0].startswith('Variant-a') and parts[1].lower() == 'test' and Path(e['path']).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            add('tomatovillage', e['path'], parts[2], 'test')
    # All source TEST photos; annotations determine image-level and object-level eligibility.
    repo, rev, entries = indices['plantdoc']
    xmls = [e for e in entries if e['path'].startswith('TEST/') and e['path'].lower().endswith('.xml')]
    def annotation(e):
        safe_name = hashlib.sha256(e['path'].encode()).hexdigest()[:20] + '.xml'
        content = fetch(raw(repo, rev, e['path']), SOURCE / 'plantdoc_annotations' / safe_name)
        return e['path'], ET.fromstring(content)
    paths = {e['path'] for e in entries}
    mapping = {'Tomato Early blight leaf': 'Early_blight', 'Tomato leaf late blight': 'Late_blight', 'Tomato leaf': 'Healthy'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for path, annotation_root in pool.map(annotation, xmls):
            objects = []
            for obj in annotation_root.findall('object'):
                name = obj.findtext('name', '').strip()
                box = obj.find('bndbox')
                if box is None: continue
                objects.append(dict(source_label=name, label=mapping.get(name, 'Other'), box=[float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
            image_path = str(Path(path).with_suffix('.jpg')).replace('\\', '/')
            if image_path not in paths or not objects: continue
            unique = {o['label'] for o in objects}
            add('plantdoc', image_path, next(iter(unique)) if len(unique) == 1 else 'Mixed', 'TEST', objects)
            size = annotation_root.find('size')
            if size is not None:
                records[-1]['annotation_size'] = [int(float(size.findtext(k, '0'))) for k in ('width', 'height')]
    # Deterministic lab-domain challenge set. No claim these were unseen during training.
    groups = {}
    for e in indices['plantvillage'][2]:
        parts = e['path'].split('/')
        if len(parts) == 4 and parts[:2] == ['raw', 'color'] and Path(e['path']).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            groups.setdefault(parts[2], []).append(e['path'])
    pv_map = {'Tomato___Early_blight': 'Early_blight', 'Tomato___Late_blight': 'Late_blight', 'Tomato___healthy': 'Healthy'}
    for category in sorted(groups):
        if not category.startswith('Tomato___'): continue
        label = pv_map.get(category, 'Other')
        n = 50 if label != 'Other' else 15
        chosen = sorted(groups[category]); random.Random(f'{SEED}:{category}').shuffle(chosen)
        for path in chosen[:n]: add('plantvillage', path, label, 'seeded_public_sample')
    print('Selected:', dict(Counter(r['source'] for r in records)), flush=True)
    failures = []
    def download(job):
        record, destination = job
        try:
            content = fetch(record['source_url'], destination)
            with Image.open(io.BytesIO(content)) as im:
                im.load(); record['width'], record['height'] = im.size
                record['pixel_sha256'] = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            record['sha256'] = hashlib.sha256(content).hexdigest()
            return record
        except Exception as exc:
            return {'id': record['id'], 'source_url': record['source_url'], 'error': str(exc)}
    success = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for index, row in enumerate(pool.map(download, jobs), 1):
            (failures if 'error' in row else success).append(row)
            if index % 50 == 0: print('Downloaded/decoded:', index, '/', len(jobs), flush=True)
    (OUT / 'images_manifest.json').write_text(json.dumps(success, indent=2), encoding='utf-8')
    (OUT / 'download_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    print('Complete:', dict(Counter(r['source'] for r in success)), 'failures:', len(failures), flush=True)


if __name__ == '__main__':
    main()
