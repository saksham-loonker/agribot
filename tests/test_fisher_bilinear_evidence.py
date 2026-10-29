import hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/training/evidence/fisher_bilinear_controls_20261029'
sha=lambda b:hashlib.sha256(b).hexdigest()

class FisherBilinearEvidenceTests(unittest.TestCase):
    def test_execution_and_hardening_have_distinct_exact_identities(self):
        m=json.loads((E/'executed_sources.json').read_text());self.assertEqual(sha((E/'executed_sources.zip').read_bytes()),m['archive_sha256'])
        with zipfile.ZipFile(E/'executed_sources.zip') as z:
            sources={r['path']:r['sha256'] for r in m['files']}
            for r in m['files']:self.assertEqual(sha(z.read(r['path'])),r['sha256'])
        for r in json.loads((E/'post_execution_hardening.json').read_text())['changes']:
            self.assertEqual(r['executed_sha256'],sources[r['path']]);self.assertEqual(r['current_lf_sha256'],sha((ROOT/r['path']).read_text(encoding='utf-8').encode()))

    def test_protocol_fold_isolation_and_original_safety_gates(self):
        m=json.loads((E/'protocols.json').read_text());self.assertEqual(sha((E/'protocols.zip').read_bytes()),m['archive_sha256']);sources={Path(r['path']).name:r['sha256'] for r in json.loads((E/'executed_sources.json').read_text())['files']}
        with zipfile.ZipFile(E/'protocols.zip') as z:
            for r in m['protocols']:
                raw=z.read(r['path']);self.assertEqual(sha(raw),r['sha256']);p=json.loads(raw);self.assertEqual(p['source_sha256'],sources[Path(r['path']).parts[0]+'.py']);self.assertTrue(p.get('original_gates_unchanged',p.get('original_safety_and_recognition_gates_unchanged',False)));self.assertFalse(p['acceptance_data_used']);self.assertEqual(p['new_encoder_inputs'],0)
                if 'calibration_folds' in p:
                    held=[];groups=p['cv_groups']
                    for f in p['calibration_folds']:
                        self.assertFalse({groups[i] for i in f['fit']}&{groups[i] for i in f['test']});held+=f['test']
                    self.assertEqual(sorted(held),list(range(len(p['train_indices']))))

    def test_recorded_results_and_selection_audit_remain_failures(self):
        files={r['path']:r['sha256'] for r in json.loads((E/'artifact_manifest.json').read_text())['files']}
        for path in E.rglob('result.json'):
            r=json.loads(path.read_text());self.assertEqual(sha(path.read_text(encoding='utf-8').encode()),files[path.relative_to(E).as_posix()]);self.assertFalse(r.get('cheap_pass',False));self.assertFalse(r.get('model_promoted',False))
            for c in r.get('controls',[]):self.assertFalse(c['cheap_pass'])
            if 'passing_policies' in r:self.assertEqual(r['passing_policies'],0)
        for path in E.rglob('checkpoint_selection_audit.json'):
            a=json.loads(path.read_text());self.assertEqual(a['corrected_selection']['epoch'],-1);self.assertTrue(a['gate_outcome_unchanged'])
        receipt=json.loads((E/'checkpoint_backup.json').read_text());self.assertEqual(receipt['verified_artifacts'],188);self.assertTrue(receipt['all_manifest_payloads_sha256_verified'])
