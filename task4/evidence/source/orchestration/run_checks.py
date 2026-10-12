"""Run the CPU semantic suite and return an auditable result and transcript."""
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def run():
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    transcript = stream.getvalue()
    return {'utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
            'passed': result.wasSuccessful() and result.testsRun == 37 and not result.skipped,
            'synthetic_fixtures_only': True, 'pretrained_model_weights_loaded': False,
            'task4_responses_generated': 0, 'task4_judgments_produced': 0, 'human_labels_assigned': 0,
            'tested_scripts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(ROOT.glob('*.py'))},
            'transcript_sha256': hashlib.sha256(transcript.encode()).hexdigest()}, transcript
