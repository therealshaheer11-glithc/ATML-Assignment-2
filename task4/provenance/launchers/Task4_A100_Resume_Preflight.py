"""Inspect a fresh A100 and compatibility with the previously approved exact pins."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
RESUME_ATTEMPT = '20261012T040418.708090Z'
RESUME_SOURCE_SHA = '818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'
LOCAL = Path('/content/task4_isolated_runtime_v1')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked evidence: '+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def verify_previous(root, helper):
    previous = root/'resume_preflight_attempts'/RESUME_ATTEMPT
    report = read(previous/'REPORT.json')
    require(report['status'] == 'SAVED_GENERATIONS_VERIFIED_RUNTIME_INVENTORIED' and
            report['script_sha256'] == RESUME_SOURCE_SHA and report['responses'] == 1800 and
            report['generation_batches'] == 452 and report['judgments'] == 0 and
            report['installs'] == report['model_loads'] == report['generation_calls'] == report['judge_calls'] == 0 and
            report['historical_contracts_rewritten'] is False, 'CPU resume evidence differs from returned result')
    for relative,digest in report['artifact_sha256'].items():
        require(helper.sha(helper.safe_file(previous,relative)) == digest,'CPU resume artifact differs: '+relative)
    for relative,digest in report['metadata_sha256'].items():
        require(helper.sha(helper.safe_file(root,relative)) == digest,'Frozen preparation contract changed: '+relative)
    require(report['source_manifest_sha256'] == helper.SOURCES,'Resume source identities differ')
    return report


def classify_base(check, selected):
    text = check.get('stdout','')+'\n'+check.get('stderr','')
    messages = sorted(line.strip() for line in text.splitlines() if line.strip())
    expected = 'ipython 7.34.0 requires jedi, which is not installed.'
    if check.get('returncode') == 0:
        return {'state':'FULL_BASE_DEPENDENCY_CHECK_CLEAN','messages':messages,'dependency_exceptions_accepted':[]}
    if check.get('returncode') == 1 and messages == [expected] and selected.get('ipython',{}).get('version') == '7.34.0' and 'jedi' not in selected:
        return {'state':'ONLY_PREVIOUSLY_APPROVED_JEDI_REPAIR_NEEDED_AFTER_ISOLATED_VALIDATION',
                'messages':messages,'repair_version':'0.20.0','repair_performed':False,'dependency_exceptions_accepted':[]}
    return {'state':'UNEXPECTED_BASE_DEPENDENCY_RESULT_REVIEW_REQUIRED','messages':messages,'dependency_exceptions_accepted':[]}


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    previous = ROOT/'resume_preflight_attempts'/RESUME_ATTEMPT
    source = previous/'RESUME_CHECK_SOURCE.py'
    require(sha(source) == RESUME_SOURCE_SHA,'Saved resume helper differs')
    helper = load('task4_a100_resume_helpers',source)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT/'resume_a100_preflight_attempts'/stamp
    attempt.mkdir(parents=True,exist_ok=False)
    report = {'format':'ATML_TASK4_RESUME_A100_PREFLIGHT_V1','attempt_utc':stamp,'status':'RUNNING',
              'source_sha256':sha(__file__),'artifact_sha256':{},'package_changes_performed':False,
              'environment_created':False,'weight_loads':0,'generation_calls':0,'judge_calls':0,
              'gpu_exceptions_accepted':[],'historical_contracts_rewritten':False,
              'wheel_bodies_rehashed_now':False,'cached_metadata_only_compatibility_check':True,
              'inference_readiness_validated':False}
    def save(name,value):
        payload = value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        path = attempt/name
        require(not path.exists(),'Refusing evidence overwrite')
        temporary = attempt/('.pending-'+name)
        with temporary.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        require(path.read_bytes() == payload,'Evidence readback failed')
        report['artifact_sha256'][name] = sha(path)
    try:
        save('PREFLIGHT_SOURCE.py',Path(__file__).read_bytes())
        print('Step 1/5: verifying the successful CPU resume report and its saved evidence.',flush=True)
        baseline = verify_previous(ROOT,helper)
        report['cpu_resume_report_sha256'] = sha(previous/'REPORT.json')
        report['reused_verified_responses'] = baseline['responses']
        report['judgments_before_this_block'] = baseline['judgments']
        folders = {}
        for index,(kind,digest) in enumerate(helper.SOURCES.items(),1):
            print(f'  Source integrity {index}/19: {kind}',flush=True)
            folder = ROOT/(kind+'_releases')/digest
            require(not folder.is_symlink() and helper.sha(folder/'MANIFEST.json') == digest,'Sealed source differs: '+kind)
            for relative,expected in read(folder/'MANIFEST.json').items():
                require(helper.sha(helper.safe_file(folder,relative)) == expected,'Source payload differs: '+kind+'/'+relative)
            folders[kind] = folder
        print('Step 2/5: checking the currently assigned A100; yesterday’s GPU UUID is historical.',flush=True)
        gates = helper.load('task4_resume_a100_gates',folders['gpu_environment']/'gates.py')
        smi = shutil.which('nvidia-smi')
        require(smi is not None,'No NVIDIA GPU found. Select an A100 runtime for this block.')
        gpu = helper.command([smi,'--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits'])
        save('GPU_QUERY.json',gpu)
        require(gpu.get('returncode') == 0,'GPU query failed; saved for review')
        report['a100_hardware'] = gates.parse_a100_inventory(gpu['stdout'])
        print(json.dumps(report['a100_hardware'],indent=2),flush=True)
        print('Step 3/5: inventorying the fresh base and running its complete dependency check.',flush=True)
        base = helper.inventory(sys.executable)
        save('BASE_INVENTORY.json',base)
        require('inventory' in base,'Fresh base metadata inventory failed')
        selected = base['inventory']['selected_packages']
        check = helper.command([sys.executable,'-I','-B','-m','pip','check'])
        save('BASE_PIP_CHECK.json',check)
        report['base_dependency_assessment'] = classify_base(check,selected)
        report['base_python'] = base['inventory']['python']
        report['base_package_names'] = len(selected)
        report['base_duplicate_groups'] = len(base['inventory']['duplicate_groups'])
        report['base_original_version_differences'] = {
            name:{'approved_original':version,'actual':selected.get(name,{}).get('version')}
            for name,version in {'huggingface-hub':'1.33.0','peft':'0.21.1','tokenizers':'0.23.2','transformers':'5.18.0'}.items()
            if selected.get(name,{}).get('version') != version}
        print('Base assessment:',json.dumps(report['base_dependency_assessment'],indent=2),flush=True)
        print('Step 4/5: testing approved cached metadata against this Python/platform; no wheel installation.',flush=True)
        from packaging.markers import default_environment
        from packaging.tags import sys_tags
        from packaging.utils import canonicalize_name
        from packaging.version import Version
        wheel_helpers = helper.load('task4_resume_wheel_guards',folders['wheel_cache']/'wheels.py')
        dependencies = helper.load('task4_resume_dependency_guards',folders['diagnostic']/'dependencies.py')
        install_guards = helper.load('task4_resume_install_guards',folders['isolated_gpu']/'guards.py')
        pins = read(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json')
        approval = read(folders['isolated_gpu']/'APPROVAL.json')
        require(approval['approved'] is True and approval['gpu_exceptions_authorized'] is False and
                sha(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json') == approval['approved_isolated_versions_sha256'],
                'Exact isolation approval differs')
        wheel_path = ROOT/'environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json'
        wheels = read(wheel_path)
        require(wheels['isolated_package_versions'] == pins and len(pins) == 84 and
                len(wheels['files']) == len(wheels['wheel_inspections']) == 90,'Approved package/cache coverage differs')
        current = default_environment()
        tags = list(sys_tags())
        for name,record in wheels['files'].items():
            require(name == record['filename'],'Wheel record name differs')
            wheel_helpers.validate_record(record,tags,current['python_full_version'])
        records = {(canonicalize_name(entry['metadata']['name']),Version(entry['metadata']['version'])):entry
                   for entry in wheels['wheel_inspections'].values()}
        course,full = wheel_helpers.check_closure(dependencies.inspect_chain,records,pins,current,
            (folders['preflight']/'course_reference/requirements.txt').read_text().splitlines(),
            {name:version for name,version in pins.items() if name != 'pip'})
        save('CURRENT_MARKER_COURSE_CHAIN.json',course)
        save('CURRENT_MARKER_ISOLATED_CHAIN.json',full)
        report['current_marker_environment'] = current
        report['historical_marker_environment'] = wheels['saved_gpu_marker_environment']
        report['marker_environment_differences'] = {key:{'historical':wheels['saved_gpu_marker_environment'].get(key),'current':value}
            for key,value in current.items() if wheels['saved_gpu_marker_environment'].get(key) != value}
        report['wheelhouse_contract_sha256'] = sha(wheel_path)
        report['compatible_cached_wheel_records'] = 90
        report['compatible_isolated_exact_pins'] = len(pins)
        report['required_course_distributions'] = len(course['nodes'])
        report['cache_metadata_closures_passed'] = True
        print('Compatible cached records: 90/90; exact pins: 84/84; course closure: 83 distributions.',flush=True)
        print('Step 5/5: checking local disk space and whether an isolated runtime already exists.',flush=True)
        report['disk_space'] = install_guards.disk_requirement(wheels['sizes']['isolated_84_wheels_zip_uncompressed_file_bytes'],shutil.disk_usage('/content').free)
        report['isolated_directory_exists'] = LOCAL.exists()
        if LOCAL.exists():
            require(LOCAL.is_dir() and not LOCAL.is_symlink(),'Existing isolated path is not an ordinary directory')
            owner = read(LOCAL/'.task4_isolation_owner.json')
            save('EXISTING_ISOLATED_OWNER.json',owner)
            report['existing_isolated_owner_requires_review_before_reuse'] = True
        report['status'] = 'A100_RESUME_INVENTORY_AND_CACHED_METADATA_COMPATIBILITY_VERIFIED'
        if report['base_original_version_differences'] or report['base_dependency_assessment']['state'] == 'UNEXPECTED_BASE_DEPENDENCY_RESULT_REVIEW_REQUIRED':
            report['status'] = 'A100_INVENTORY_SAVED_BASE_REVIEW_REQUIRED'
        report['next_stage'] = 'Review returned runtime evidence; restore the approved exact isolated stack with fresh wheel-byte verification, then validate it before judging.'
    except Exception as error:
        import traceback
        report['status'] = 'A100_RESUME_PREFLIGHT_STOPPED_REVIEW_REQUIRED'
        report['error'] = {'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    # REPORT must not hash itself.
    final_path = attempt/'REPORT.json'
    payload = (json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temporary = attempt/'.pending-REPORT.json'
    with temporary.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    require(not final_path.exists(),'Refusing final report overwrite')
    os.replace(temporary,final_path)
    require(final_path.read_bytes() == payload,'Final report readback failed')
    print('Saved report:',final_path,flush=True)
    print('Status:',report['status'],flush=True)
    for key in ['base_python','base_package_names','base_duplicate_groups','base_original_version_differences',
                'marker_environment_differences','isolated_directory_exists','disk_space']:
        if key in report:print(key+':',json.dumps(report[key],sort_keys=True),flush=True)
    if 'error' in report:print(json.dumps(report['error'],indent=2),flush=True)
    print('Responses retained: last verified 1800/1800; judgments: 0/1800; human labels last confirmed: 0/240.',flush=True)
    print('No packages installed or changed; no model loads, new generation or judging. Return this full output.',flush=True)


if __name__ == '__main__':
    main()
