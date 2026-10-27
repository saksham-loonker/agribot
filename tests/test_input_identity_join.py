"""Caches with overlapping local indices must never reuse a different image."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"model_training/v3"))
from input_identity_join import input_identity_join


class InputIdentityJoinTests(unittest.TestCase):
    def test_scene_and_new_image_same_index_do_not_match(self):
        old=[dict(input_sha256="a"*64),dict(input_sha256="b"*64)]
        appended=[dict(input_sha256="a"*64),dict(input_sha256="c"*64)]
        self.assertEqual(input_identity_join(appended,old,[0,1]),[0,None])

    def test_reordered_sparse_and_repeated_inputs_join_by_bytes(self):
        old=[dict(input_sha256=key*64) for key in "abcd"]
        target=[dict(input_sha256=key*64) for key in "dbdca"]
        self.assertEqual(input_identity_join(target,old,[3,1,0]),[0,1,0,None,2])

    def test_missing_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            input_identity_join([dict(input_sha256="not-a-sha")],[],[])


if __name__=="__main__": unittest.main()
