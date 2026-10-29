"""A newly admitted publisher must not disappear into a legacy domain bucket."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
from source_balance import source_class_group_weights


class SourceBalanceTests(unittest.TestCase):
    def test_duplicate_views_and_large_publishers_do_not_change_source_mass(self):
        rows = [
            dict(y=y, source=source, group=group)
            for y in (0, 1)
            for source, families, views in (("legacy", 20, 3), ("new_ad", 2, 1))
            for group in range(families)
            for _ in range(views)
        ]
        values = source_class_group_weights(rows, list(range(len(rows))))
        for label in (0, 1):
            for source in ("legacy", "new_ad"):
                self.assertAlmostEqual(
                    float(
                        sum(
                            w
                            for r, w in zip(rows, values)
                            if r["y"] == label and r["source"] == source
                        )
                    ),
                    0.25,
                )
        np.testing.assert_allclose(values.sum(), 1.0)

    def test_partition_reordering_retains_sample_weights(self):
        rows = [
            dict(y=0, source="a", group=1),
            dict(y=1, source="a", group=1),
            dict(y=1, source="b", group=2),
        ]
        np.testing.assert_allclose(
            source_class_group_weights(rows, [2, 0, 1]),
            source_class_group_weights(rows, [0, 1, 2])[[2, 0, 1]],
        )

    def test_single_class_publisher_does_not_break_global_class_balance(self):
        rows = [
            dict(y=0, source="a", group=1),
            dict(y=1, source="a", group=1),
            dict(y=1, source="b", group=2),
        ]
        np.testing.assert_allclose(
            source_class_group_weights(rows, [0, 1, 2]), [0.5, 0.25, 0.25]
        )

    def test_empty_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            source_class_group_weights([], [])


if __name__ == "__main__":
    unittest.main()
