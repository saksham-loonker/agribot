import hashlib,json,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


class AuxiliaryEvidenceTests(unittest.TestCase):
    def test_public_archives_and_independent_backup_receipts(self):
        for folder,count in [('attribution_controls_20261029',170),('auxiliary_controls_20261029',228)]:
            root=ROOT/'docs/training/evidence'/folder
            for name,key in [('executed_sources','files'),('protocols','protocols')]:
                manifest=json.loads((root/(name+'.json')).read_text())
                self.assertEqual(hashlib.sha256((root/(name+'.zip')).read_bytes()).hexdigest(),manifest['archive_sha256'])
                with zipfile.ZipFile(root/(name+'.zip')) as z:
                    self.assertIsNone(z.testzip())
                    for r in manifest[key]:self.assertEqual(hashlib.sha256(z.read(r['path'])).hexdigest(),r['sha256'])
            backup=json.loads((root/'checkpoint_backup.json').read_text())
            self.assertEqual(backup['verified_artifacts'],count)
            self.assertTrue(backup.get('all_member_sha256_verified',backup.get('all_manifest_payloads_sha256_verified',False)))

    def test_absent_fallback_is_not_zero_classifier_accuracy(self):
        root=ROOT/'docs/training/evidence/auxiliary_controls_20261029/phase235_scope_attribution'
        result=json.loads((root/'result.json').read_text())
        self.assertTrue(result['invalid_as_whole_primary_model_test'])
        invalid=root/'invalid_initial_attribution/INVALID.json'
        self.assertTrue(invalid.exists())

    def test_completed_coverage_fits_keep_holdouts_separate(self):
        root=ROOT/'docs/training/evidence/coverage_controls_20261029'
        manifest=json.loads((root/'protocols.json').read_text())
        self.assertEqual(hashlib.sha256((root/'protocols.zip').read_bytes()).hexdigest(),manifest['archive_sha256'])
        with zipfile.ZipFile(root/'protocols.zip') as z:
            for item in manifest['protocols']:
                raw=z.read(item['path']);self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])
                protocol=json.loads(raw)
                if 'calibration_folds' not in protocol:continue
                train=set(protocol['train_indices']);validation=set(protocol['validation_indices']);held=[]
                self.assertFalse(train&validation)
                for fold in protocol['calibration_folds']:
                    fit=set(fold['fit']);test=set(fold['test']);self.assertFalse(fit&test)
                    self.assertEqual(fit|test,train);self.assertFalse((fit|test)&validation);held.extend(test)
                self.assertEqual(sorted(held),sorted(train))
        for phase in ['phase245_all_photo_nonlinear_supervision','phase247_all_photo_suffix_adaptation']:
            result=json.loads((root/phase/'result.json').read_text())
            self.assertFalse(result['model_promoted'])
            for control in result['controls']:self.assertFalse(control['cheap_pass'])

    def test_prefix_writer_repair_keeps_executed_identity_explicit(self):
        root=ROOT/'docs/training/evidence/coverage_controls_20261029'
        sources={r['path']:r['sha256'] for r in json.loads((root/'executed_sources.json').read_text())['files']}
        for change in json.loads((root/'post_execution_hardening.json').read_text())['changes']:
            self.assertEqual(sources[change['path']],change['executed_sha256'])
            self.assertEqual(hashlib.sha256((ROOT/change['path']).read_text().encode()).hexdigest(),change['current_lf_sha256'])
        repair=json.loads((root/'phase246_whole_prefix_coverage/cache_format_repair.json').read_text())
        self.assertTrue(repair['payload_bytes_unchanged']);self.assertEqual(repair['shape'],[234,256,768])
