import json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from phase242_training_coverage_audit import audit


class TrainingCoverageAuditTests(unittest.TestCase):
    def test_missing_roi_and_shared_validation_identity_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def put(name,value):
                target=p/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value))
            rows=[dict(path='val',source='olid_tomato',label='Healthy',group='v',validation=True),
                  dict(path='missing',source='olid_tomato',label='Healthy',group='m',validation=False),
                  dict(path='shared',source='olid_tomato',label='Healthy',group='s',validation=False),
                  dict(path='safe',source='olid_tomato',label='Healthy',group='t',validation=False)]
            put('phase84_population_head/fitting_rows.json',rows)
            put('phase127_clinical_anchors/protocol.json',dict(train_indices=[3],quarantined_overlap_groups=['s']))
            put('phase87_diagnostic_bag_features/bags.json',{name:dict(center=[0],full=[0]) for name in ['val','shared']}|{'safe':dict(center=[1],full=[1])})
            put('phase29_compact/inputs.json',[dict(input_sha256='validation-pixel'),dict(input_sha256='training-pixel')])
            put('phase87_diagnostic_bag_features/new_inputs.json',[])
            put('phase29_compact/source_files.json',{r['path']:r['path']+'-file' for r in rows})
            put('phase24_olid_source/sealed_manifest.json',[])
            result=audit(p);cases={r['index']:r for r in result['cases']}
            self.assertEqual(cases[1]['reasons'],['no_cached_detected_roi'])
            self.assertIn('validation_roi_input',cases[2]['reasons'])
            self.assertIn('quarantined_overlap_family',cases[2]['reasons'])
            self.assertTrue(cases[3]['selected']);self.assertFalse(cases[3]['reasons'])
            self.assertFalse(result['acceptance_data_used']);self.assertFalse(result['model_promoted'])
