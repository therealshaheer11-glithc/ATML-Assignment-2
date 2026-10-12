# Task 4 read-only resume check: no installations, model loads, or inference.
SOURCES = {'preflight': '9a121f29750b5b09c5feafeb491630ba4b88b6c91dc2ed5ee94ebdb989793b65', 'base_layout_diagnostic': '961b9825e7a34823a8f3930e597149c425427d494f8d9c9b10652e5998a734be', 'base_repair_recheck': '4955a6f76dfbfc5b2f92825fce309eabbe4c209cf576b73a7d182ff0b101f09a', 'base_repair': '76963d33df89cd6439d988bd11808dd503a3724bde6d5c5ef02dec172037b47b', 'diagnostic': '71bdb7fdbe130c5adaef8328437371bcd89621d0a93e849bab61b7db8491c540', 'environment': '5767d9553b4b4c32f6470cbd59cc7afa4ae6cc56be838b9be8f0f85ce0f47ff1', 'generation': '862f0e0330d9e8e571cb02bc27b32c48591aaee15f9f2543d315dc43ce97ada4', 'gpu_environment': 'd941ecb6c16ec91a942b319167826007f5124b0dedc10d2a337b3b783ff4015b', 'isolated_gpu': '7a5673003062448f66a3c43763340483968234f42d2659d517c47d7e33de6e8a', 'isolated_recheck': '999b26996654587e6ddfda1a146cac4ac38b421f0856f6134abd2ac61433c363', 'isolation_preflight': '2ddda0858de0b95a43e006f77b56e569cf037beeec97c06e3f8b3a99275435d5', 'judge_download': '0f1b51a93ec1d861f336fe4eae1c360939e166526eceeaeab023e6b5ac7d1f16', 'model_inputs': '50696b4becd13a8f1594896d384e718fa8d3984e4fb5785bec977a6b35eeac7d', 'model_inventory': 'd9e3eb2cd1d9c5cafc4956313553209d0581a07ed9d204c92529be56755e22de', 'model_loading': '20c7fb542b21e7d41afd36e2f2b2bb4fa6eb981994eda20ae1344bac4c06935d', 'orchestration': '35ca690761d2bfda0992ff37855b16fc72c03ffd4b47afaa81edc533ab7a5f65', 'post_restart': 'b05d32bf434b196d2b01b3477319fbdb32d71cd33f281d57c820e8a61b7577cc', 'recovery': '6629e696e04632096dcd79e3f41fe1e28fb55e7aee8c95ec17e33f383928dda8', 'wheel_cache': 'df469ec50f8fc5e493f2f818eb490858c713eb88e3216309aeb43e5fb49a7aab'}
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import traceback

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL = Path('/content/task4_isolated_runtime_v1')
POLICIES = ['sft', 'dpo', 'ppo', 'grpo']
GENERATION_ATTEMPT = '20261011T174101.363739Z'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_file(folder, relative):
    require(isinstance(relative, str), 'Artifact name is not text')
    name = PurePosixPath(relative)
    require(relative not in {'', '.', '..'} and not name.is_absolute() and
            '..' not in name.parts and '\\' not in relative, 'Unsafe artifact path')
    path = Path(folder)/relative
    require(path.is_file() and not path.is_symlink() and
            path.resolve().is_relative_to(Path(folder).resolve()), 'Artifact escaped folder: '+relative)
    return path


def verify_sources(root):
    folders = {}
    for kind, expected in SOURCES.items():
        folder = root/(kind+'_releases')/expected
        require(not folder.is_symlink(), 'Linked source folder')
        require(sha(folder/'MANIFEST.json') == expected, 'Sealed source differs: '+kind)
        for relative, digest in read(folder/'MANIFEST.json').items():
            require(sha(safe_file(folder, relative)) == digest, 'Source payload differs: '+kind+'/'+relative)
        folders[kind] = folder
    return folders


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def check_saved(root, folders):
    core = load('task4_resume_core', folders['orchestration']/'core.py')
    io = load('task4_resume_generation_io', folders['generation']/'generation_io.py')
    attempt = root/'generation_attempts'/GENERATION_ATTEMPT
    previous = read(attempt/'REPORT.json')
    require(previous['status'] == 'ALL_FOUR_POLICIES_GENERATED_AND_SAVED' and
            previous['source_manifest_sha256'] == SOURCES['generation'], 'Generation success evidence differs')
    for relative, digest in previous['artifact_sha256'].items():
        require(sha(safe_file(attempt, relative)) == digest, 'Generation attempt artifact differs: '+relative)
    completion_path = root/'GENERATION_COMPLETION_CONTRACT.json'
    require(sha(completion_path) == previous['generation_completion_contract_sha256'], 'Completion contract differs')
    completion = read(completion_path)
    require(completion['format'] == 'ATML_TASK4_GENERATION_COMPLETE_V1' and
            completion['source_manifest_sha256'] == SOURCES['generation'] and
            completion['responses'] == 1800 and completion['generation_batches'] == 452 and
            completion['policy_response_counts'] == {p:450 for p in POLICIES}, 'Completion counts differ')
    contract_path = root/'EVALUATION_EXECUTION_CONTRACT.json'
    require(sha(contract_path) == completion['evaluation_contract_sha256'] == previous['evaluation_contract_sha256'],
            'Evaluation contract differs')
    require((root/'evaluation_v1/CONTRACT.json').read_bytes() == contract_path.read_bytes(), 'Stored evaluation identity differs')
    contract = read(contract_path)
    inputs = read(root/'INPUT_IDENTITY_CONTRACT.json')['inputs']['xstest']
    require(sha(inputs['path']) == inputs['sha256'], 'XSTest file differs')
    rows = io.load_rows(inputs['path'], core)
    require(core.validate_rows(rows) == inputs['ordered_prompt_ids'], 'XSTest source order differs')
    protocol = read(root/'CPU_ORCHESTRATION_VALIDATION_CONTRACT.json')
    io.verify_rows(rows, protocol['fixed_evaluation_protocol'], core)
    evidence_files = {
        'cpu_model_input_contract_sha256':'CPU_MODEL_INPUT_CONTRACT.json',
        'cpu_orchestration_contract_sha256':'CPU_ORCHESTRATION_VALIDATION_CONTRACT.json',
        'gpu_environment_contract_sha256':'GPU_ENVIRONMENT_CONTRACT.json',
        'model_loading_validation_contract_sha256':'MODEL_LOADING_VALIDATION_CONTRACT.json',
        'input_identity_contract_sha256':'INPUT_IDENTITY_CONTRACT.json',
        'audit_id_contract_sha256':'AUDIT_ID_CONTRACT.json'}
    evidence = {key:sha(root/name) for key,name in evidence_files.items()}
    approval = folders['preflight']/'docs/approval.json'
    require(read(approval)['approved'] is True and read(root/'OWNER.json')['approval_sha256'] == sha(approval), 'Task approval differs')
    for name, digest in protocol['course_helper_sha256'].items():
        require(sha(safe_file(folders['preflight']/'course_reference', name)) == digest, 'Course helper differs: '+name)
    evidence.update(task4_approval_sha256=sha(approval), course_helper_sha256=protocol['course_helper_sha256'],
                    orchestration_source_manifest_sha256=SOURCES['orchestration'])
    model_inputs = read(root/'CPU_MODEL_INPUT_CONTRACT.json')
    eos = model_inputs['tokenizer_and_generation_metadata']['policy']['tokenizer']['eos_token_id']
    require(contract == core.make_contract(rows, SOURCES['generation'], evidence, eos), 'Frozen evaluation protocol/evidence differs')
    # UnitStore only reads here: its immutable constructor returns for byte-identical existing evidence.
    require((root/'evaluation_v1/CONTRACT.json').read_bytes() == core.encoded(contract), 'Contract canonical bytes differ')
    store = core.UnitStore(root/'evaluation_v1', contract)
    expected_units = {f'generation/{p}/{i:06d}.json' for p in POLICIES for i in range(113)}
    require(set(completion['generation_unit_sha256']) == expected_units, 'Generation unit coverage differs')
    for name, digest in completion['generation_unit_sha256'].items():
        require(sha(safe_file(root/'evaluation_v1', name)) == digest, 'Generation unit bytes differ: '+name)
    require(set(completion['exports']) == set(POLICIES), 'Export policy coverage differs')
    counts = {}
    for policy in POLICIES:
        batches = core.saved_generations(store, policy, rows, require_complete=True)
        generated = [row for unit in range(113) for row in batches[unit]]
        export = root/'evaluation_v1/exports/generation'/(policy+'.jsonl')
        require(sha(export) == completion['exports'][policy]['sha256'] and
                export.read_bytes() == io.jsonl_bytes(generated), 'Export differs from saved response records: '+policy)
        core.require(len(generated) == 450, 'Response coverage differs')
        store.check_paths('judgment', policy, [row['xstest_id'] for row in generated])
        judgments = 0
        for row in generated:
            unit, digest = row['xstest_id'], core.checksum(row)
            payload = store.read('judgment', policy, unit, digest)
            if payload is not None:
                require(set(payload) == {'xstest_id','policy','generation_sha256','judge'} and
                        payload['xstest_id'] == unit and payload['policy'] == policy and
                        payload['generation_sha256'] == digest, 'Cached judgment identity differs')
                core.validate_judge_output(payload['judge'])
                judgments += 1
        counts[policy] = {'responses':len(generated), 'batches':len(batches), 'judgments':judgments}
        print(f'{policy}: {len(generated)}/450 responses; {len(batches)}/113 batches; {judgments}/450 judgments', flush=True)
    return counts, core, model_inputs


INVENTORY_CODE = r'''
import importlib.metadata as m, json, platform, re, site, sys
from pathlib import Path
entries = {}
for d in m.distributions():
    name=d.metadata.get('Name')
    if name:
        key=re.sub(r'[-_.]+','-',name).lower()
        entries.setdefault(key,[]).append({'version':d.version,'root':str(Path(d.locate_file('')).resolve()),
            'metadata_path':str(getattr(d,'_path',''))})
selected={}
for name in sorted(entries):
    d=m.distribution(name)
    selected[name]={'version':d.version,'root':str(Path(d.locate_file('')).resolve()),
                    'metadata_path':str(getattr(d,'_path',''))}
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'executable':sys.executable,
    'prefix':sys.prefix,'base_prefix':sys.base_prefix,'user_site_enabled':site.ENABLE_USER_SITE,
    'sys_path':sys.path,'selected_packages':selected,'duplicate_groups':{k:v for k,v in entries.items() if len(v)>1}},indent=2))
'''


def command(args, timeout=120):
    try:
        result = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=timeout)
        return {'args':[str(x) for x in args], 'returncode':result.returncode,
                'stdout':result.stdout, 'stderr':result.stderr}
    except Exception as error:
        return {'args':[str(x) for x in args], 'error':type(error).__name__+': '+str(error)}


def inventory(python):
    result = command([python,'-I','-B','-c',INVENTORY_CODE])
    if result.get('returncode') == 0:
        result['inventory'] = json.loads(result.pop('stdout'))
    return result


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    require(ROOT.is_dir(), 'Saved Task 4 experiment directory is missing')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT/'resume_preflight_attempts'/stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'format':'ATML_TASK4_RESUME_PREFLIGHT_V1','saved_utc':stamp,
              'script_sha256':sha(__file__), 'status':'RESUME_CHECK_IN_PROGRESS',
              'installs':0,'model_loads':0,'generation_calls':0,'judge_calls':0,
              'historical_contracts_rewritten':False,'current_environment_validated_for_inference':False,
              'model_and_wheel_bytes_rehashed_now':False,'artifact_sha256':{}}
    core = None
    def save(name, value):
        path = attempt/name
        payload = value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
        require(not path.exists(), 'Refusing report overwrite')
        temporary = attempt/('.pending-'+name)
        with temporary.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        require(path.read_bytes() == payload, 'Report readback failed')
        return sha(path)
    try:
        report['artifact_sha256']['RESUME_CHECK_SOURCE.py'] = save('RESUME_CHECK_SOURCE.py',Path(__file__).read_bytes())
        folders = verify_sources(ROOT)
        report['source_manifest_sha256'] = SOURCES
        print('All 19 saved source releases verified.',flush=True)
        counts, core, model_inputs = check_saved(ROOT,folders)
        report['counts_by_policy'] = counts
        report['responses'] = sum(c['responses'] for c in counts.values())
        report['generation_batches'] = sum(c['batches'] for c in counts.values())
        report['judgments'] = sum(c['judgments'] for c in counts.values())
        report['judgments_remaining'] = 1800-report['judgments']
        report['human_labels'] = {'last_confirmed':0,'required':240,'annotations_checked_by_this_block':False}
        metadata_names = ['OWNER.json','INPUT_IDENTITY_CONTRACT.json','AUDIT_ID_CONTRACT.json',
            'CPU_MODEL_INPUT_CONTRACT.json','CPU_ORCHESTRATION_VALIDATION_CONTRACT.json',
            'BASE_MODEL_REFERENCE.json','JUDGE_SNAPSHOT_CONTRACT.json','GPU_ENVIRONMENT_CONTRACT.json',
            'ISOLATED_GPU_ENVIRONMENT_CONTRACT.json','ISOLATED_CHILD_ENVIRONMENT_CONTRACT.json',
            'COLAB_BASE_REPAIR_CONTRACT.json','MODEL_LOADING_VALIDATION_CONTRACT.json',
            'EVALUATION_EXECUTION_CONTRACT.json','GENERATION_COMPLETION_CONTRACT.json',
            'environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json']
        report['metadata_sha256'] = {name:sha(ROOT/name) for name in metadata_names}
        # Model/wheel existence and sizes are an inventory, not fresh byte-integrity validation.
        for name in ['BASE_MODEL_REFERENCE.json','JUDGE_SNAPSHOT_CONTRACT.json']:
            model = read(ROOT/name)
            for filename, entry in model['verified_files'].items():
                path = safe_file(Path(model['directory']),filename)
                require(path.stat().st_size == entry['bytes'], 'Model file size differs: '+filename)
        wheelhouse = ROOT/'environment_wheelhouse'
        wheels = read(wheelhouse/'VERIFIED_WHEELHOUSE_CONTRACT.json')
        require(len(wheels['verified_files']) == 90, 'Approved wheel count differs')
        require(wheels['isolated_package_versions'] == read(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json'),
                'Cached isolated pins differ from the approved source')
        for filename,entry in wheels['verified_files'].items():
            require(safe_file(wheelhouse/'wheels',filename).stat().st_size == entry['bytes'], 'Wheel file size differs: '+filename)
        for role,entry in model_inputs['prepared_adapters'].items():
            require(role in {'dpo','ppo','grpo'} and Path(entry['directory']) == ROOT/'prepared_adapters'/role, 'Adapter location differs')
            for filename,digest in entry['files_sha256'].items():
                require(sha(safe_file(Path(entry['directory']),filename)) == digest, 'Adapter bytes differ: '+role+'/'+filename)
        require(set(model_inputs['prepared_adapters']) == {'dpo','ppo','grpo'}, 'Adapter coverage differs')
        report['model_wheel_sizes_and_adapter_hashes_checked'] = True
        receipts = []
        for path in sorted((ROOT/'resume_receipts').glob('*.json')):
            receipt = read(path)
            receipts.append({'path':str(path),'sha256':sha(path),'status':receipt.get('status'),
                             'saved_utc':receipt.get('saved_utc')})
        report['prior_break_receipts'] = receipts
        print('Prior break receipts found:', len(receipts), '(generation was verified independently).',flush=True)
        print('Inspecting current interpreters and dependencies; no package changes.',flush=True)
        base = inventory(sys.executable)
        report['artifact_sha256']['BASE_INVENTORY.json'] = save('BASE_INVENTORY.json',base)
        require('inventory' in base, 'Current base interpreter inventory failed; inspect saved diagnostic')
        base_check = command([sys.executable,'-I','-B','-m','pip','check'])
        report['artifact_sha256']['BASE_PIP_CHECK.json'] = save('BASE_PIP_CHECK.json',base_check)
        base_data = base.get('inventory',{})
        report['base_python'] = base_data.get('python')
        report['base_key_packages'] = {name:base_data.get('selected_packages',{}).get(name)
            for name in ['torch','transformers','tokenizers','peft','huggingface-hub','trl',
                         'bitsandbytes','numpy','pip','ipython','jedi','parso','gradio','diffusers']}
        report['base_package_names'] = len(base_data.get('selected_packages',{}))
        report['base_duplicate_groups'] = len(base_data.get('duplicate_groups',{}))
        report['base_pip_check_returncode'] = base_check.get('returncode')
        report['isolated_directory_exists'] = LOCAL.exists()
        if LOCAL.exists():
            require(LOCAL.is_dir() and not LOCAL.is_symlink(), 'Isolated runtime path is not an ordinary directory')
            owner = read(LOCAL/'.task4_isolation_owner.json')
            require(owner['source_manifest_sha256'] == SOURCES['isolated_gpu'] and
                    owner['directory'] == str(LOCAL) and owner['system_site_packages'] is False and
                    owner['wheelhouse_contract_sha256'] == sha(wheelhouse/'VERIFIED_WHEELHOUSE_CONTRACT.json') and
                    owner['approval_sha256'] == read(ROOT/'ISOLATED_GPU_ENVIRONMENT_CONTRACT.json')['approval_sha256'],
                    'Existing local interpreter ownership differs; preserved for review')
            report['artifact_sha256']['CURRENT_ISOLATED_OWNER.json'] = save('CURRENT_ISOLATED_OWNER.json',owner)
            python = LOCAL/'bin/python'
            require(python.is_file(), 'Existing isolated interpreter is missing')
            isolated = inventory(python)
            report['artifact_sha256']['ISOLATED_INVENTORY.json'] = save('ISOLATED_INVENTORY.json',isolated)
            iso_check = command([python,'-I','-B','-m','pip','check'])
            report['artifact_sha256']['ISOLATED_PIP_CHECK.json'] = save('ISOLATED_PIP_CHECK.json',iso_check)
            expected = wheels['isolated_package_versions']
            selected = {name:entry['version'] for name,entry in isolated.get('inventory',{}).get('selected_packages',{}).items()}
            report['isolated_pin_differences'] = {name:{'expected':expected.get(name),'actual':selected.get(name)}
                                                 for name in sorted(set(expected)|set(selected)) if expected.get(name)!=selected.get(name)}
            report['isolated_pip_check_returncode'] = iso_check.get('returncode')
        smi = shutil.which('nvidia-smi')
        gpu = command([smi,'--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader']) if smi else {'available':False}
        report['artifact_sha256']['GPU_INVENTORY.json'] = save('GPU_INVENTORY.json',gpu)
        report['gpu_inventory'] = gpu
        report['status'] = 'SAVED_GENERATIONS_VERIFIED_RUNTIME_INVENTORIED'
        report['next_stage'] = 'Review this current runtime inventory, then restore/validate the approved isolated environment before judging.'
    except Exception as error:
        report['status'] = 'RESUME_CHECK_STOPPED_EVIDENCE_PRESERVED'
        report['error'] = {'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    save('REPORT.json',report)
    print('Saved report:',attempt/'REPORT.json',flush=True)
    print('Status:',report['status'],flush=True)
    if report['status'] != 'SAVED_GENERATIONS_VERIFIED_RUNTIME_INVENTORIED':
        print(json.dumps(report['error'],indent=2),flush=True)
        raise RuntimeError('Resume verification stopped. Return the complete output before any restoration or judging.')
    print('Verified responses:',str(report['responses'])+'/1800; remaining: 0',flush=True)
    print('Verified judgments:',str(report['judgments'])+'/1800; remaining:',report['judgments_remaining'],flush=True)
    print('Human labels: last confirmed 0/240; this block does not inspect human annotations.',flush=True)
    print('Base Python:',report['base_python'],'; package names:',report['base_package_names'],
          '; duplicate groups:',report['base_duplicate_groups'],'; pip check return code:',report['base_pip_check_returncode'],flush=True)
    print('Base dependency check:',base_check.get('stdout',''),base_check.get('stderr',''),base_check.get('error',''),flush=True)
    print('Selected base package versions:',json.dumps({name:entry['version'] if entry else None
        for name,entry in report['base_key_packages'].items()},sort_keys=True),flush=True)
    print('Isolated directory exists:',report['isolated_directory_exists'],flush=True)
    if report['isolated_directory_exists']:
        print('Isolated pin differences:',json.dumps(report['isolated_pin_differences'],sort_keys=True),flush=True)
        print('Isolated dependency check:',iso_check.get('returncode'),iso_check.get('stdout',''),iso_check.get('stderr',''),iso_check.get('error',''),flush=True)
    print('GPU inventory:',json.dumps(gpu,indent=2),flush=True)
    print('No installation, model loading, generation or judging performed. Return this complete output.',flush=True)


if __name__ == '__main__':
    main()
