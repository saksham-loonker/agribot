"""Verify committed repair evidence remains bound to exact executed inputs."""
import hashlib,json,unittest,zipfile
from pathlib import Path

EVIDENCE=Path(__file__).resolve().parents[1]/'docs/training/evidence/recovery_controls_20261028'
def sha(value):return hashlib.sha256(value).hexdigest()

class ConvergenceEvidenceTests(unittest.TestCase):
    def test_executed_source_archive_hashes(self):
        manifest=json.loads((EVIDENCE/'executed_sources.json').read_text())
        archive=EVIDENCE/'executed_sources.zip'
        self.assertEqual(sha(archive.read_bytes()),manifest['archive_sha256'])
        with zipfile.ZipFile(archive) as z:
            for row in manifest['files']:
                self.assertTrue(row['path'].startswith('model_training/v3/'))
                self.assertEqual(sha(z.read(row['path'])),row['sha256'],row['path'])

    def test_lossless_protocol_archive_and_execution_sources(self):
        manifest=json.loads((EVIDENCE/'extended_protocols_183_211.json').read_text())
        archive=EVIDENCE/'extended_protocols_183_211.zip'
        self.assertEqual(sha(archive.read_bytes()),manifest['archive_sha256'])
        sources={Path(row['path']).name:row['sha256'] for row in json.loads((EVIDENCE/'executed_sources.json').read_text())['files']}
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(set(z.namelist()),{row['path'] for row in manifest['protocols']})
            for row in manifest['protocols']:
                value=z.read(row['path']);self.assertEqual(sha(value),row['sha256']);self.assertEqual(len(value),row['bytes'])
                protocol=json.loads(value);script=Path(row['path']).parent.name+'.py'
                if script in sources:self.assertEqual(protocol['source_sha256'],sources[script],script)

    def test_results_remain_bound_to_archived_protocols(self):
        manifest=json.loads((EVIDENCE/'extended_protocols_183_211.json').read_text())
        for row in manifest['protocols']:
            folder=EVIDENCE/Path(row['path']).parent
            for result_file in folder.rglob('result.json'):
                result=json.loads(result_file.read_text())
                if 'protocol_sha256' in result:self.assertEqual(result['protocol_sha256'],row['sha256'],str(result_file))
                self.assertFalse(result.get('model_promoted',False),str(result_file))
