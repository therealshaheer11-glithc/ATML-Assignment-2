from pathlib import Path
import tempfile
import unittest
from . import core, preflight
from .diagnostics import score_one, controlled_pairs, summarize
from .test_core import synthetic_diagnostics
from task5_feedback import rlvr


class PreviousCheckpointTests(unittest.TestCase):
    def fixture(self, project):
        rows = synthetic_diagnostics()
        data = project / 'diagnostics.jsonl'
        data.write_bytes(b''.join(core.json_bytes(r) for r in rows))
        previous = project / 'prior'
        previous.mkdir()
        scores = [score_one(r, i, rlvr) for i, r in enumerate(rows)]
        pairs = controlled_pairs(scores)
        summary = summarize(scores, pairs)
        for name, value in [('EXACT_SCORES.json', scores), ('VERIFIER_CONTROLLED_PAIRS.json', pairs),
                            ('VERIFIER_SUMMARY.json', summary)]:
            core.write_once(previous / name, value)
        manifest = {p.name: core.file_sha(p) for p in previous.iterdir()}
        core.write_once(previous / 'EVIDENCE_MANIFEST.json', manifest)
        core.write_once(previous / 'REPORT.json', {
            'status': 'CPU_EXACT_DIAGNOSTICS_SAVED_REVIEW_PENDING',
            'diagnostic_scores_saved': 100, 'diagnostic_scores_computed': 100,
            'unique_verifier_pairs': 80, 'errors': [],
            'evidence_manifest_sha256': core.file_sha(previous / 'EVIDENCE_MANIFEST.json')})
        return {'diagnostics': 'diagnostics.jsonl',
                'block6_report': {'path': 'prior/REPORT.json', 'sha256': core.file_sha(previous / 'REPORT.json')}}

    def test_previous_100_scores_and_80_pairs_reverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            bindings = self.fixture(project)
            result = preflight.verify_previous_scores(project, bindings)
            self.assertEqual(result['scores_verified'], 100)
            self.assertEqual(result['controlled_pairs_verified'], 80)

    def test_changed_saved_evidence_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            bindings = self.fixture(project)
            (project / 'prior/EXACT_SCORES.json').write_text('[]')
            with self.assertRaisesRegex(RuntimeError, 'evidence differs'):
                preflight.verify_previous_scores(project, bindings)


if __name__ == '__main__':
    unittest.main()
