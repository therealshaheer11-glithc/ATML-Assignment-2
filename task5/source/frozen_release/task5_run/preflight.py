"""One consolidated CPU pass; no installs, pretrained models or AI inference."""
from datetime import datetime, timezone
import importlib.metadata
import io
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from . import core
from .run import inputs, verify_source


def inventory():
    return sorted((d.metadata.get('Name', ''), d.version, str(d.locate_file('')))
                  for d in importlib.metadata.distributions())


def checked_json(path, expected):
    core.require(path.is_file() and not path.is_symlink() and core.file_sha(path) == expected,
                 'Saved checkpoint identity differs: ' + str(path))
    return core.read_json(path)


def helper(release, name):
    source = release / 'vendor' / name
    module = types.ModuleType('task5_' + name.replace('.', '_'))
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
    return module


def verify_snapshots(project, bindings):
    verified = {}
    for role, spec in bindings['snapshots'].items():
        folder = project / spec['path']
        core.require(folder.is_dir() and not folder.is_symlink(), 'Model snapshot directory differs.')
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                  if p.is_file() and '.cache' not in p.relative_to(folder).parts}
        core.require(actual == set(spec['files']), 'Model snapshot file coverage differs: ' + role)
        for index, (name, expected) in enumerate(spec['files'].items(), 1):
            path = folder / name
            print(f'  {role}: file {index}/{len(spec["files"])}: {name}', flush=True)
            core.require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes']
                         and core.file_sha(path) == expected['sha256'], 'Model snapshot differs: ' + name)
            verified[role + '/' + name] = expected
    return verified


def verify_wheels(project, release, bindings):
    from packaging.markers import default_environment
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    cache_path = project / bindings['wheel_contract']['path']
    cache = checked_json(cache_path, bindings['wheel_contract']['sha256'])
    pins = core.read_json(release / 'package_versions.json')
    core.require(cache['isolated_package_versions'] == pins and len(pins) == 84
                 and len(cache['files']) == 90, 'Approved wheel set differs.')
    wheels, dependencies = helper(release, 'wheels.py'), helper(release, 'dependencies.py')
    environment, tags = default_environment(), list(sys_tags())
    core.require(environment['implementation_name'] == 'cpython' and environment['python_version'] == '3.13'
                 and environment['platform_system'] == 'Linux' and environment['platform_machine'] == 'x86_64',
                 'Fresh runtime platform differs from the approved wheel candidate.')
    directory = cache_path.parent / 'wheels'
    core.require({p.name for p in directory.iterdir()} == set(cache['files']), 'Wheel cache file coverage differs.')
    nodes, verified = {}, {}
    for i, (name, record) in enumerate(sorted(cache['files'].items()), 1):
        print(f'  wheel {i}/90: {name}', flush=True)
        value = wheels.verify_file(directory / name, record)
        core.require(value == cache['verified_files'][name], 'Wheel receipt identity differs.')
        inspected = wheels.inspect_wheel(directory / name, record, tags, environment['python_full_version'])
        core.require(inspected == cache['wheel_inspections'][name], 'Wheel metadata changed.')
        if pins.get(canonicalize_name(record['package'])) == record['version']:
            key = canonicalize_name(record['package'])
            core.require(key not in nodes, 'Duplicate selected wheel package.')
            nodes[key] = inspected['metadata']
        verified[name] = value['sha256']
    full = dependencies.inspect_chain([f'{n}=={v}' for n, v in sorted(pins.items())], nodes.get, environment)
    course = dependencies.inspect_chain((release / 'requirements.txt').read_text().splitlines(), nodes.get, environment)
    core.require(full['required_chain_passed'] and course['required_chain_passed']
                 and {k: v['version'] for k, v in full['nodes'].items()} == pins
                 and {k: v['version'] for k, v in course['nodes'].items()} == {k: v for k, v in pins.items() if k != 'pip'},
                 'Approved dependency closure no longer passes.')
    return {'verified_wheels': verified, 'isolated_packages': len(full['nodes']),
            'course_packages': len(course['nodes']), 'current_platform': environment,
            'dependency_checks_passed': True}


def verify_previous_scores(project, bindings):
    from .diagnostics import group_rows, score_one, controlled_pairs, summarize
    from task5_feedback import rlvr
    previous = project / bindings['block6_report']['path']
    report = checked_json(previous, bindings['block6_report']['sha256'])
    core.require(report['status'] == 'CPU_EXACT_DIAGNOSTICS_SAVED_REVIEW_PENDING'
                 and report['diagnostic_scores_saved'] == report['diagnostic_scores_computed'] == 100
                 and report['unique_verifier_pairs'] == 80 and not report['errors'],
                 'Last successful exact-scoring checkpoint differs.')
    manifest = checked_json(previous.parent / 'EVIDENCE_MANIFEST.json', report['evidence_manifest_sha256'])
    for name, expected in manifest.items():
        path = previous.parent / name
        core.require(path.resolve().is_relative_to(previous.parent.resolve()), 'Unsafe prior evidence path.')
        core.require(path.is_file() and not path.is_symlink() and core.file_sha(path) == expected,
                     'Previously saved exact-scoring evidence differs: ' + name)
    rows = core.read_rows(project / bindings['diagnostics'])
    group_rows(rows)
    scores = [score_one(row, i, rlvr) for i, row in enumerate(rows)]
    pairs = controlled_pairs(scores)
    summary = summarize(scores, pairs)
    core.require(scores == core.read_json(previous.parent / 'EXACT_SCORES.json')
                 and pairs == core.read_json(previous.parent / 'VERIFIER_CONTROLLED_PAIRS.json')
                 and summary == core.read_json(previous.parent / 'VERIFIER_SUMMARY.json'),
                 'Recomputed CPU verifier results differ from the saved checkpoint.')
    return {'scores_verified': 100, 'controlled_pairs_verified': 80,
            'report_sha256': core.file_sha(previous), 'evidence_files_verified': len(manifest)}


def main(project=Path('/content/drive/MyDrive/ATML-Assignment-2')):
    import os
    smi = shutil.which('nvidia-smi')
    gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                         text=True, timeout=15) if smi else None
    core.require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or
                      os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
                 'Use CPU / None for this consolidated preflight.')
    release = Path(__file__).resolve().parents[1]
    root = project / 'task5/experiment_v1'
    core.require(root.is_dir(), 'Mount the existing Drive project first.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = root / 'combined_cpu_preflight_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'errors': [], 'installs': 0, 'downloads': 0,
              'pretrained_model_loads': 0, 'policy_answers_generated': 0, 'ai_judgments': 0}
    before = inventory()
    try:
        print('1/5: verifying the approved protocol and packaged source.', flush=True)
        release_sha = verify_source(release)
        protocol = core.read_json(release / 'protocol.json')
        core.require(protocol['approval_status'] == 'APPROVED' and protocol['approved_proposal_sha256'] ==
                     'e020dc3d0f29d766953fd56dbfd49ae75b2d34378ebcb87878556e1e03fa0a8e',
                     'Approved protocol identity differs.')
        b, rows = inputs(project, release)
        report['release_sha256'] = release_sha
        report['generation_units'] = len(core.generation_schedule(rows))
        print('2/5: verifying last night\'s 100 scores and 80 verifier comparisons.', flush=True)
        report['prior_exact_scoring'] = verify_previous_scores(project, b)
        print('3/5: checking the 16 saved model files; loading no models.', flush=True)
        report['verified_snapshot_files'] = verify_snapshots(project, b)
        print('4/5: checking all 90 cached wheels and the exact 84-package plan.', flush=True)
        report['wheel_validation'] = verify_wheels(project, release, b)
        print('5/5: running the complete CPU implementation test suite.', flush=True)
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            ['task5_run.test_core', 'task5_run.test_inference', 'task5_run.test_orchestration',
             'task5_run.test_preflight'])
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (attempt / 'CPU_TESTS.txt').write_text(log.getvalue())
        report['tests_run'], report['tests_passed'] = result.testsRun, result.wasSuccessful()
        core.require(result.wasSuccessful(), 'CPU implementation tests failed; inspect saved test log.')
        core.require(before == inventory(), 'Base package inventory changed during checks.')
        contract = {'complete': True, 'release_sha256': release_sha,
                    'protocol_sha256': core.file_sha(release / 'protocol.json'),
                    'fixed_asset_sha256': b['asset_sha256'], 'verified_snapshot_files': report['verified_snapshot_files'],
                    'verified_wheels': report['wheel_validation']['verified_wheels'],
                    'isolated_packages': 84, 'course_packages': 83,
                    'tests_passed': True, 'tests_run': result.testsRun,
                    'prior_exact_report_sha256': b['block6_report']['sha256'],
                    'planned_policy_answers': 1200, 'planned_ai_comparisons': 880,
                    'a100_setup_and_synthetic_cuda_checks_pending': True,
                    'pinned_tokenizer_and_real_model_checks_pending': True,
                    'pretrained_execution_separate_approval_required': True}
        target = root / 'CPU_READINESS_CONTRACT.json'
        if target.exists():
            core.require(core.read_json(target) == contract, 'Existing readiness contract differs; preserve and review.')
        else:
            core.write_once(target, contract)
        report['cpu_readiness_sha256'] = core.file_sha(target)
        report['status'] = 'CPU_READY_FOR_APPROVED_A100_SETUP'
    except Exception as exc:
        report['status'] = 'COMBINED_CPU_PREFLIGHT_STOPPED_REVIEW_PENDING'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        report['base_package_inventory_unchanged'] = before == inventory()
        if not report['base_package_inventory_unchanged']:
            report['status'] = 'COMBINED_CPU_PREFLIGHT_STOPPED_REVIEW_PENDING'
            report['errors'].append('Base package inventory changed.')
        core.write_once(attempt / 'REPORT.json', report)
        print('Status:', report['status'])
        print('Previous exact scores verified:', report.get('prior_exact_scoring'))
        print('Model files verified:', len(report.get('verified_snapshot_files', {})), '/ 16')
        print('Wheels verified:', len(report.get('wheel_validation', {}).get('verified_wheels', {})), '/ 90')
        print('CPU tests:', report.get('tests_run'), 'passed:', report.get('tests_passed'))
        print('Remaining workload: 1,200 policy answers; 880 AI comparison events.')
        print('Errors:', report['errors'])
        print('Saved report:', attempt / 'REPORT.json')
        print('Report SHA-256:', core.file_sha(attempt / 'REPORT.json'))
        print('0 installs; 0 downloads; 0 pretrained model loads; 0 new policy answers; 0 AI judgments.')
        print('Send this output before the A100 step.')
    return report


if __name__ == '__main__':
    raise SystemExit(0 if main()['status'] == 'CPU_READY_FOR_APPROVED_A100_SETUP' else 1)
