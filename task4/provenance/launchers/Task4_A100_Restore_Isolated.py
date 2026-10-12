"""Restore only the previously approved exact isolated stack on the current A100."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL = Path('/content/task4_isolated_runtime_v1')
PREFLIGHT_ATTEMPT = '20261012T042038.114257Z'
PREFLIGHT_SOURCE_SHA = 'b304191c43f791b5c9b742758389027b7c7c0de034d72ed7881ad25747d5a657'
RESUME_ATTEMPT = '20261012T040418.708090Z'
RESUME_SOURCE_SHA = '818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'


def require(value,message):
    if not value:
        raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Missing or linked evidence: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return module


def verify_preflight(root,helper):
    attempt=root/'resume_a100_preflight_attempts'/PREFLIGHT_ATTEMPT
    previous=read(attempt/'REPORT.json')
    require(previous['status']=='A100_RESUME_INVENTORY_AND_CACHED_METADATA_COMPATIBILITY_VERIFIED' and
            previous['source_sha256']==PREFLIGHT_SOURCE_SHA and previous['compatible_cached_wheel_records']==90 and
            previous['compatible_isolated_exact_pins']==84 and previous['required_course_distributions']==83 and
            previous['cache_metadata_closures_passed'] is True and previous['marker_environment_differences']=={} and
            previous['base_original_version_differences']=={} and previous['gpu_exceptions_accepted']==[] and
            previous['package_changes_performed'] is False and previous['environment_created'] is False and
            previous['weight_loads']==previous['generation_calls']==previous['judge_calls']==0,
            'A100 preflight differs from returned successful evidence')
    for relative,digest in previous['artifact_sha256'].items():
        require(sha(helper.safe_file(attempt,relative))==digest,'A100 preflight artifact differs: '+relative)
    require(sha(attempt/'PREFLIGHT_SOURCE.py')==PREFLIGHT_SOURCE_SHA,'A100 preflight source differs')
    return previous,attempt


def owned_environment(local,source_sha,approval_sha,wheelhouse_sha,preflight_sha,python_version,hardware):
    return {'format':'ATML_TASK4_RESUMED_LOCAL_ISOLATION_OWNER_V1','directory':str(local),
        'source_sha256':source_sha,'approval_sha256':approval_sha,'wheelhouse_contract_sha256':wheelhouse_sha,
        'resume_preflight_report_sha256':preflight_sha,'base_python_executable':str(Path(sys.executable).resolve()),
        'target_python_full_version':python_version,'system_site_packages':False,
        'hardware_gpu_uuids':[entry['uuid'] for entry in hardware]}


def check_existing_owner(local,expected):
    require(not local.is_symlink(),'Local isolated runtime path is linked')
    if local.exists():
        require(local.is_dir(),'Local isolated runtime path is not a directory')
        owner=local/'.task4_isolation_owner.json'
        require(owner.is_file() and not owner.is_symlink() and read(owner)==expected,
                'Existing local interpreter is not this approved owned restored environment; preserved for review')


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
    helper_path=ROOT/'resume_preflight_attempts'/RESUME_ATTEMPT/'RESUME_CHECK_SOURCE.py'
    require(sha(helper_path)==RESUME_SOURCE_SHA,'Pinned resume source differs')
    helper=load('task4_restore_resume_helpers',helper_path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt=ROOT/'resume_isolated_restore_attempts'/stamp
    attempt.mkdir(parents=True,exist_ok=False)
    report={'format':'ATML_TASK4_RESUMED_ISOLATED_RESTORE_V1','attempt_utc':stamp,'status':'RUNNING',
        'source_sha256':sha(__file__),'artifact_sha256':{},'commands':[],
        'scope':'Existing approved exact 84-package isolated A100 restoration only; base repair follows separately.',
        'wheel_downloads':0,'base_package_changes_performed':False,'historical_contracts_rewritten':False,
        'pretrained_model_loads':0,'task4_generation_calls':0,'task4_judge_calls':0,'training_performed':False,
        'gpu_exceptions_accepted':[],'base_repair_and_judge_model_validation_pending':True,
        'fresh_wheels_verified':0,'isolated_installation_attempted':False}
    def save(name,value):
        payload=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        path=attempt/name;path.parent.mkdir(parents=True,exist_ok=True)
        require(not path.exists() and not path.is_symlink(),'Refusing evidence overwrite: '+name)
        temporary=path.parent/('.pending-'+path.name)
        with temporary.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        require(not path.exists(),'Concurrent writer created evidence')
        os.replace(temporary,path);require(path.read_bytes()==payload,'Evidence readback failed')
        if name!='REPORT.json':report['artifact_sha256'][name]=sha(path)
    def command(args,label,environment):
        args=[str(x) for x in args]
        event={'stage':label,'argv':args,'child_environment_policy':'Approved isolation sanitation; recognized notebook MPLBACKEND removed without replacement.'}
        save(label+'_COMMAND.json',event)
        print('Runtime preparation:',label,flush=True)
        log_path=attempt/(label+'.log')
        with log_path.open('x') as log:
            process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,env=environment,cwd='/content')
            try:
                for line in process.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            log.flush();os.fsync(log.fileno())
        event.update(returncode=code,log_sha256=sha(log_path))
        report['artifact_sha256'][log_path.name]=event['log_sha256']
        save(label+'_RESULT.json',event);report['commands'].append(event)
        require(code==0,label+' failed; evidence preserved, with no automatic fallback')
    try:
        save('RESTORE_SOURCE.py',Path(__file__).read_bytes())
        print('Step 1/6: verifying saved A100/CPU preflight evidence and unchanged approved sources.',flush=True)
        previous,previous_folder=verify_preflight(ROOT,helper)
        preflight_module=helper.load('task4_restore_preflight_helpers',previous_folder/'PREFLIGHT_SOURCE.py')
        cpu=preflight_module.verify_previous(ROOT,helper)
        require(sha(ROOT/'resume_preflight_attempts'/RESUME_ATTEMPT/'REPORT.json')==previous['cpu_resume_report_sha256'],
                'Bound CPU resume report changed')
        report['resume_preflight_report_sha256']=sha(previous_folder/'REPORT.json')
        folders={}
        for index,(kind,digest) in enumerate(helper.SOURCES.items(),1):
            print(f'  Source integrity {index}/19: {kind}',flush=True)
            folder=ROOT/(kind+'_releases')/digest
            require(not folder.is_symlink() and sha(folder/'MANIFEST.json')==digest,'Sealed source differs: '+kind)
            for relative,expected in read(folder/'MANIFEST.json').items():
                require(sha(helper.safe_file(folder,relative))==expected,'Source payload differs: '+kind+'/'+relative)
            folders[kind]=folder
        from packaging.markers import default_environment
        from packaging.tags import sys_tags
        from packaging.utils import canonicalize_name
        from packaging.version import Version
        wheels=helper.load('wheels',folders['wheel_cache']/'wheels.py')
        cache=helper.load('task4_restore_cache_plan',folders['wheel_cache']/'cache_wheels.py')
        guards=helper.load('task4_restore_original_guards',folders['isolated_gpu']/'guards.py')
        isolation=helper.load('task4_restore_child_policy',folders['isolated_recheck']/'isolation.py')
        dependencies=helper.load('task4_restore_dependencies',folders['diagnostic']/'dependencies.py')
        gates=helper.load('task4_restore_a100_gates',folders['gpu_environment']/'gates.py')
        child_env=isolation.child_environment(os.environ,guards.clean_environment)
        parent_backend=os.environ.get('MPLBACKEND')
        hardware_query=helper.command(['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits'])
        save('CURRENT_GPU_QUERY.json',hardware_query)
        require(hardware_query.get('returncode')==0,'Current GPU query failed')
        hardware=gates.parse_a100_inventory(hardware_query['stdout'])
        require(hardware==previous['a100_hardware'],'A100 changed since this session preflight; recheck before installation')
        target=default_environment()
        require(target==previous['current_marker_environment'],'Current Python/platform changed since preflight')
        before=helper.inventory(sys.executable);save('BASE_BEFORE_INVENTORY.json',before)
        require('inventory' in before and before==read(previous_folder/'BASE_INVENTORY.json'),
                'Fresh base metadata/layout changed since preflight')
        approval_path=folders['isolated_gpu']/'APPROVAL.json'
        approval=read(approval_path)
        require(approval['approved'] is True and approval['gpu_exceptions_authorized'] is False and
                'isolated A100 environment preserving all 84 reviewed pins' in approval['approved_scope'] and
                sha(folders['isolated_gpu']/'APPROVED_PROPOSAL.md')==approval['approved_proposal_sha256'],
                'Approved restoration scope differs')
        report['approval_sha256']=sha(approval_path)
        pins=read(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json')
        require(sha(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json')==approval['approved_isolated_versions_sha256'],
                'Exact approved pins differ')
        wheelhouse=ROOT/'environment_wheelhouse';wheel_contract_path=wheelhouse/'VERIFIED_WHEELHOUSE_CONTRACT.json'
        wheel_contract=read(wheel_contract_path)
        require(sha(wheel_contract_path)==previous['wheelhouse_contract_sha256']==cpu['metadata_sha256']['environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json'],
                'Approved wheel contract changed')
        require(wheel_contract['approval_sha256']==report['approval_sha256'] and wheel_contract['isolated_package_versions']==pins and
                wheel_contract['all_wheels_verified'] is True and
                wheel_contract['cache_identity_contract_sha256']==sha(wheelhouse/'CACHE_IDENTITY_CONTRACT.json'),
                'Wheel cache approval/provenance differs')
        wheel_attempt=ROOT/'wheel_cache_attempts/20261011T151647.989109Z'
        plan_path=wheel_attempt/'APPROVED_ISOLATION_PROPOSAL.json'
        require(sha(plan_path)==wheel_contract['isolation_proposal_sha256'],'Approved wheel proposal differs')
        tags=list(sys_tags())
        files=cache.validate_plan(read(plan_path),pins,tags,target['python_full_version'])
        require(files==wheel_contract['files'] and len(files)==90,'Exact approved file records differ')
        wheel_dir=wheelhouse/'wheels';receipts=wheelhouse/'receipts'
        require(not wheelhouse.is_symlink() and not wheel_dir.is_symlink() and not receipts.is_symlink(),
                'Wheelhouse directories are linked')
        require({p.name for p in wheel_dir.iterdir()}==set(files) and
                {p.name for p in receipts.iterdir()}=={name+'.json' for name in files},'Wheel/receipt coverage differs')
        print('Step 2/6: fresh checksum and metadata checks of all 90 cached wheels; no downloads.',flush=True)
        measurements={}
        for index,(name,record) in enumerate(files.items(),1):
            print(f'  Cached wheel {index}/90: {name}',flush=True)
            result=wheels.verify_file(wheel_dir/name,record)
            require(result==wheel_contract['verified_files'][name]==read(receipts/(name+'.json')),'Wheel receipt differs: '+name)
            inspected=wheels.inspect_wheel(wheel_dir/name,record,tags,target['python_full_version'])
            require(inspected==wheel_contract['wheel_inspections'][name],'Wheel metadata differs: '+name)
            measurements[name]=inspected
            save('wheel_checks/'+name+'.json',{'verified_file':result,'metadata_matches_approved_contract':True})
            report['fresh_wheels_verified']=index
        records={(canonicalize_name(entry['metadata']['name']),Version(entry['metadata']['version'])):entry for entry in measurements.values()}
        course,full=wheels.check_closure(dependencies.inspect_chain,records,pins,target,
            (folders['preflight']/'course_reference/requirements.txt').read_text().splitlines(),
            {name:version for name,version in pins.items() if name!='pip'})
        save('FRESH_WHEEL_COURSE_CHAIN.json',course);save('FRESH_WHEEL_FULL_CHAIN.json',full)
        space=guards.disk_requirement(wheel_contract['sizes']['isolated_84_wheels_zip_uncompressed_file_bytes'],shutil.disk_usage('/content').free)
        save('LOCAL_INSTALLATION_SPACE.json',space)
        owner=owned_environment(LOCAL,report['source_sha256'],report['approval_sha256'],sha(wheel_contract_path),
            report['resume_preflight_report_sha256'],target['python_full_version'],hardware)
        check_existing_owner(LOCAL,owner)
        # All approval, wheel-byte, compatibility, hardware and base guards precede mutation.
        print('Step 3/6: creating or resuming only the approved owned isolated environment.',flush=True)
        if not LOCAL.exists():
            LOCAL.mkdir()
            payload=(json.dumps(owner,indent=2,sort_keys=True)+'\n').encode()
            owner_file=LOCAL/'.task4_isolation_owner.json'
            with owner_file.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            require(owner_file.read_bytes()==payload,'Local ownership readback differs')
        save('LOCAL_OWNER.json',owner)
        if not (LOCAL/'pyvenv.cfg').exists():
            venv.EnvBuilder(with_pip=False,system_site_packages=False,clear=False,symlinks=True).create(LOCAL)
        python=LOCAL/'bin/python'
        require(python.is_file(),'Isolated interpreter was not created')
        inspect=folders['isolated_gpu']/'inspect_python.py'
        command([python,'-I','-B',inspect,attempt/'ISOLATED_BEFORE.json'],'isolated_inventory_before',child_env)
        installed=guards.validate_inventory(read(attempt/'ISOLATED_BEFORE.json'),pins,LOCAL)
        require(read(attempt/'ISOLATED_BEFORE.json')['python']==target['python_full_version'],'Isolated Python differs')
        save('OFFLINE_HASHED_REQUIREMENTS.txt',guards.requirement_lines(files,pins,wheel_dir).encode())
        print('Step 4/6: restoring the exact 84 approved pins offline with hashes.',flush=True)
        if 'pip' not in installed:
            save('OFFLINE_PIP_BOOTSTRAP.txt',guards.requirement_lines(files,{'pip':pins['pip']},wheel_dir).encode())
            pip_record=next(record for record in files.values() if canonicalize_name(record['package'])=='pip' and record['version']==pins['pip'])
            bootstrap_env=isolation.child_environment(os.environ,lambda original:guards.clean_environment(original,wheel_dir/pip_record['filename']))
            report['isolated_installation_attempted']=True
            command(guards.install_command(python,attempt/'OFFLINE_PIP_BOOTSTRAP.txt',attempt/'PIP_BOOTSTRAP_INSTALL_REPORT.json',bootstrap=True),
                'bootstrap_exact_approved_pip',bootstrap_env)
        if installed!=pins:
            report['isolated_installation_attempted']=True
            command(guards.install_command(python,attempt/'OFFLINE_HASHED_REQUIREMENTS.txt',attempt/'ISOLATED_INSTALL_REPORT.json'),
                'install_exact_approved_pins',child_env)
        command([python,'-I','-B',inspect,attempt/'ISOLATED_AFTER.json'],'isolated_inventory_after',child_env)
        after_inventory=read(attempt/'ISOLATED_AFTER.json')
        guards.validate_inventory(after_inventory,pins,LOCAL,complete=True)
        command([python,'-I','-B','-m','pip','--disable-pip-version-check','check'],'isolated_full_dependency_check',child_env)
        print('Step 5/6: preserved fresh imports, tiny CUDA/NF4 probes and all 37 course checks.',flush=True)
        probe_dir=attempt/'fresh_probe';probe_dir.mkdir()
        command([python,'-I','-B','-u',folders['gpu_environment']/'probe_gpu.py',folders['preflight'],
            folders['orchestration'],folders['diagnostic'],probe_dir],'preserved_fresh_a100_probe',child_env)
        probe=read(probe_dir/'GPU_PROBE.json')
        require(probe['all_required_package_versions']=={name:version for name,version in pins.items() if name!='pip'} and
                probe['required_distributions']==83 and probe['synthetic_semantic_checks_passed']==37 and
                probe['required_dependency_chain_passed'] and probe['fresh_process_imports_passed'] and
                probe['tiny_float16_cuda_check_passed'] and probe['tiny_nf4_cuda_backend_check_passed'] and
                probe['pretrained_model_weights_loaded'] is False,'Preserved isolated validation is incomplete')
        for path in sorted(probe_dir.iterdir()):report['artifact_sha256'][str(path.relative_to(attempt))]=sha(path)
        for name in ['PIP_BOOTSTRAP_INSTALL_REPORT.json','ISOLATED_INSTALL_REPORT.json','ISOLATED_BEFORE.json','ISOLATED_AFTER.json']:
            if (attempt/name).exists():report['artifact_sha256'][name]=sha(attempt/name)
        print('Step 6/6: checking that the Colab base and historical generation evidence remain unchanged.',flush=True)
        after=helper.inventory(sys.executable);save('BASE_AFTER_INVENTORY.json',after)
        require(after==before and os.environ.get('MPLBACKEND')==parent_backend,'Base packages/layout or parent backend changed')
        for relative,digest in cpu['metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT,relative))==digest,'Historical preparation evidence changed: '+relative)
        base_check=helper.command([sys.executable,'-I','-B','-m','pip','check'])
        save('BASE_DEPENDENCY_CHECK_UNREPAIRED.json',base_check)
        require(base_check==read(previous_folder/'BASE_PIP_CHECK.json'),'Unrepaired base dependency result changed unexpectedly')
        proof={'format':'ATML_TASK4_RESUMED_ISOLATED_VALIDATION_V1','source_sha256':report['source_sha256'],
            'approval_sha256':report['approval_sha256'],'resume_preflight_report_sha256':report['resume_preflight_report_sha256'],
            'wheelhouse_contract_sha256':sha(wheel_contract_path),'local_owner_sha256':sha(LOCAL/'.task4_isolation_owner.json'),
            'environment_directory':str(LOCAL),'python_executable':str(python),'python':after_inventory['python'],
            'isolated_package_versions':pins,'hardware':hardware,'isolated_full_dependency_check_passed':True,
            'fresh_imports_cuda_nf4_passed':True,'synthetic_checks_passed':37,'fresh_cached_wheels_verified':90,
            'base_package_inventory_and_layout_unchanged':True,'historical_metadata_sha256':cpu['metadata_sha256'],
            'historical_evaluation_contract_sha256':cpu['metadata_sha256']['EVALUATION_EXECUTION_CONTRACT.json'],
            'base_repair_pending':True,'judge_model_loading_validation_pending':True,'gpu_exceptions_accepted':[]}
        save('ISOLATED_VALIDATION_CONTRACT.json',proof)
        report.update(status='RESUMED_ISOLATED_A100_VALIDATED_BASE_REPAIR_PENDING',
            isolated_distribution_count=84,required_distribution_count=83,synthetic_checks_passed=37,
            isolated_full_dependency_check_passed=True,fresh_imports_cuda_nf4_passed=True,
            base_package_inventory_and_layout_unchanged=True,
            isolated_validation_contract_sha256=sha(attempt/'ISOLATED_VALIDATION_CONTRACT.json'),
            local_free_bytes_after=shutil.disk_usage('/content').free)
    except BaseException as error:
        import traceback
        report['status']='RESUMED_ISOLATED_RESTORE_STOPPED_EVIDENCE_PRESERVED'
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    # Include partial diagnostics too, so an interrupted installation remains reviewable.
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name!='REPORT.json':
            report['artifact_sha256'][str(path.relative_to(attempt))]=sha(path)
    save('REPORT.json',report)
    print('Saved report:',attempt/'REPORT.json',flush=True)
    print('Status:',report['status'],flush=True)
    print('Fresh cached wheels verified:',str(report['fresh_wheels_verified'])+'/90',flush=True)
    if report['status']=='RESUMED_ISOLATED_A100_VALIDATED_BASE_REPAIR_PENDING':
        print('Exact isolated packages: 84/84; full dependency check PASS; imports/CUDA/NF4 PASS; synthetic checks: 37/37.',flush=True)
        print('Colab base and historical generation contracts unchanged. Base Jedi repair and judge loading remain pending.',flush=True)
    else:print(json.dumps(report.get('error'),indent=2),flush=True)
    print('Responses retained: 1800/1800; judgments: 0/1800; human labels last confirmed: 0/240.',flush=True)
    print('No pretrained model loading, new generation, judging or training. Keep this A100 connected and return the complete output.',flush=True)


if __name__=='__main__':main()
