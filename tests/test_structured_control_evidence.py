import hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/training/evidence/structured_controls_20261029'
sha=lambda b:hashlib.sha256(b).hexdigest()

class StructuredEvidenceTests(unittest.TestCase):
    def test_exact_sources_and_protocols_are_bound(self):
        sources={}
        for stem,field in [('executed_sources','files'),('protocols','protocols')]:
            m=json.loads((E/(stem+'.json')).read_text());self.assertEqual(sha((E/(stem+'.zip')).read_bytes()),m['archive_sha256'])
            with zipfile.ZipFile(E/(stem+'.zip')) as z:
                for r in m[field]:
                    raw=z.read(r['path']);self.assertEqual(sha(raw),r['sha256'])
                    if stem=='executed_sources':sources[Path(r['path']).name]=r['sha256']
                    else:
                        p=json.loads(raw);self.assertEqual(p['source_sha256'],sources[Path(r['path']).parts[0]+'.py']);self.assertEqual(p['new_encoder_inputs'],0);self.assertFalse(p['acceptance_data_used'])
                        if 'calibration_folds' in p:
                            held=[]
                            for f in p['calibration_folds']:
                                self.assertFalse({p['cv_groups'][i] for i in f['fit']}&{p['cv_groups'][i] for i in f['test']});held+=f['test']
                            self.assertEqual(sorted(held),list(range(len(p['train_indices']))));self.assertTrue(p['original_gates_unchanged']);self.assertEqual(p['C'],1.)

    def test_original_descriptor_reuse_and_private_backup(self):
        c=json.loads((E/'phase223_structured_leaf_cache/complete.json').read_text());self.assertEqual(c['shape'],[2248,144]);self.assertEqual(c['old_global_rows_reused'],2106);self.assertTrue(c['old_global_rows_bitwise_equal']);self.assertEqual(c['new_encoder_inputs'],0)
        r=json.loads((E/'checkpoint_backup.json').read_text());self.assertEqual(r['verified_artifacts'],12);self.assertTrue(r['all_manifest_payloads_sha256_verified'])

    def test_failed_controls_have_exact_result_hashes(self):
        files={r['path']:r['sha256'] for r in json.loads((E/'artifact_manifest.json').read_text())['files']}
        for path in E.rglob('result.json'):
            r=json.loads(path.read_text());self.assertEqual(sha(path.read_text(encoding='utf-8').encode()),files[path.relative_to(E).as_posix()]);self.assertFalse(r['model_promoted']);self.assertFalse(r['acceptance_data_used'])
            if 'passing_policies' in r:self.assertEqual(r['passing_policies'],0)
            for c in r.get('controls',[]):self.assertFalse(c['cheap_pass'])
