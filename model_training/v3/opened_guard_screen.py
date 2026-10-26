"""Unchanged app gates shared by bounded opened-data guard experiments."""

from cache_first_screen import retention_failures
from phase17_context_select import coarse_counts
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics


class OpenedGuardScreen:
    def __init__(self, data):
        self.data = data
        self.baseline = grouped(data["base"])
        self.sources = sorted({r["source"] for r in data["internal"]})
        self.internal_before = {
            s: photo_metrics([r for r in data["internal"] if r["source"] == s])
            for s in self.sources
        }
        self.coarse_before = coarse_counts(data["base"])

    def evaluate(self, route):
        revised = {k: [route(r) for r in rows] for k, rows in self.data["base"].items()}
        metrics = grouped(revised)
        after = [route(r) for r in self.data["internal"]]
        internal = {
            s: photo_metrics([r for r in after if r["source"] == s])
            for s in self.sources
        }
        coarse = coarse_counts(revised)
        checks = dict(
            old_retention=retain(metrics, self.baseline),
            old_field_healthy=all(
                metrics["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
                for s in ("tom2024", "swtt5wy2pr")
            ),
            coarse_retention=all(
                v[1] <= self.coarse_before[k][1] + int(0.005 * v[0])
                for k, v in coarse.items()
            ),
            internal_retention=retain(internal, self.internal_before),
            internal_healthy_each_source=all(
                m["healthy"]["correct_rate"] >= 0.7
                for m in internal.values()
                if m["healthy"]["n"]
            ),
            original_disease_and_healthy_exact=all(
                a["app_replay"] == b["app_replay"]
                for k in revised
                for a, b in zip(self.data["base"][k], revised[k])
                if a["app_replay"]["kind"] in ("HEALTHY", "DISEASE")
            ),
        )
        recalls = [
            metrics["phase5_selection"]["by_source"][s]["healthy_recall"]
            for s in ("tom2024", "swtt5wy2pr")
        ] + [
            m["healthy"]["correct_rate"] for m in internal.values() if m["healthy"]["n"]
        ]
        return dict(
            checks=checks,
            passed=all(checks.values()),
            metrics=metrics,
            internal_metrics=internal,
            worst_healthy=min(recalls),
            mean_healthy=sum(recalls) / len(recalls),
            old_failures=retention_failures(metrics, self.baseline),
            internal_failures=retention_failures(internal, self.internal_before),
        )


def ranking(candidate):
    checks = candidate["checks"]
    return (
        candidate["passed"],
        checks["old_retention"]
        and checks["coarse_retention"]
        and checks["internal_retention"],
        candidate["worst_healthy"],
        candidate["mean_healthy"],
    )
