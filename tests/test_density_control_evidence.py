import hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/training/evidence/density_controls_20261029'
def sha(b):return hashlib.sha256(b).hexdigest()

class DensityEvidenceTests(unittest.TestCase):
    def test_source_archive_and_explicit_post_execution_fixes(self):
        m=json.loads((E/'executed_sources.json').read_text());archive=E/'executed_sources.zip';self.assertEqual(sha(archive.read_bytes()),m['archive_sha256'])
        with zipfile.ZipFile(archive) as z:
            source={r['path']:r['sha256'] for r in m['files']}
            for row in m['files']:self.assertEqual(sha(z.read(row['path'])),row['sha256'])
        changes=json.loads((E/'post_execution_hardening.json').read_text())['changes']
        for row in changes:
            self.assertEqual(source[row['path']],row['executed_sha256'])
            self.assertEqual(sha((ROOT/row['path']).read_text(encoding='utf-8').encode()),row['current_source_lf_sha256'])

    def test_protocols_sources_and_calibration_folds_remain_bound(self):
        m=json.loads((E/'protocols.json').read_text());archive=E/'protocols.zip';self.assertEqual(sha(archive.read_bytes()),m['archive_sha256'])
        sources={Path(r['path']).name:r['sha256'] for r in json.loads((E/'executed_sources.json').read_text())['files']}
        with zipfile.ZipFile(archive) as z:
            for row in m['protocols']:
                raw=z.read(row['path']);self.assertEqual(sha(raw),row['sha256']);p=json.loads(raw)
                self.assertEqual(p['source_sha256'],sources[Path(row['path']).parts[0]+'.py'])
                for field,name in [('density_source_sha256','shrinkage_density.py'),('meta_loss_source_sha256','meta_domain_loss.py'),('basis_source_sha256','density_stack.py'),('data_source_sha256','exhaustive_replay_data.py')]:
                    if field in p:self.assertEqual(p[field],sources[name])
                if 'train_indices' in p:self.assertFalse(set(p['train_indices'])&set(p['validation_indices']))
                if 'calibration_folds' in p:
                    groups=p['cv_groups'];held=[]
                    for fold in p['calibration_folds']:
                        self.assertFalse(set(fold['fit'])&set(fold['test']))
                        self.assertFalse({groups[i] for i in fold['fit']}&{groups[i] for i in fold['test']});held.extend(fold['test'])
                    self.assertEqual(sorted(held),list(range(len(p['train_indices']))))
                    self.assertTrue(all(w>0 for w in p['training_weights']))
                self.assertTrue(p['original_gates_unchanged']);self.assertFalse(p['acceptance_data_used']);self.assertEqual(p['new_encoder_inputs'],0)

    def test_failed_results_cannot_become_release_evidence(self):
        m=json.loads((E/'protocols.json').read_text())
        artifacts={r['path']:r['sha256'] for r in json.loads((E/'artifact_manifest.json').read_text())['files']}
        for row in m['protocols']:
            for path in (E/Path(row['path']).parent).rglob('result.json'):
                # Git may check text out with CRLF; PC execution JSON uses LF.
                self.assertEqual(sha(path.read_text(encoding='utf-8').encode()),artifacts[path.relative_to(E).as_posix()])
                r=json.loads(path.read_text());self.assertEqual(r['protocol_sha256'],row['sha256']);self.assertFalse(r.get('model_promoted',False));self.assertFalse(r.get('acceptance_data_used',False))
                if 'passing_policies' in r:self.assertEqual(r['passing_policies'],0)
                for control in r.get('controls',[]):self.assertFalse(control.get('cheap_pass',False))
