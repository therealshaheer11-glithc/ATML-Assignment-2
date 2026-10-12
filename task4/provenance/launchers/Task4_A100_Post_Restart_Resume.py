"""Validate the current restarted notebook and both interpreters without inference."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL=Path('/content/task4_isolated_runtime_v1')
REPAIR_ATTEMPT='20261012T045114.280111Z'
REPAIR_SOURCE_SHA='6b1813443b7b61fe9be1bc0ee36e6ac4f73dcba660533d15d884aab4580f95af'
RESUME_SOURCE_SHA='818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'
RESTORE_SOURCE_SHA='a3025083ed0094652e39e306c7d2bdaf226d025a651e705cf3f8f37b17ac130e'


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Missing or linked evidence: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return module


def kernel_guard_view(contract):
    require(contract['base_kernel_restart_required'] is True and contract['gpu_exceptions_accepted']==[],
            'Saved restart requirement/exception state differs')
    return {'base_kernel_restart_required':True,'python':contract['python'],
        'effective_metadata_versions':{name:entry['version'] for name,entry in contract['base_selected_metadata'].items()}}


def observe_kernel():
    import IPython,jedi
    names={item.name for item in jedi.Script("task4_text = 'verified'\ntask4_text.up").complete(2,len('task4_text.up'))}
    observed={}
    for module_name,package_name in [('IPython','ipython'),('jedi','jedi'),('parso','parso'),('torch','torch'),
        ('numpy','numpy'),('packaging','packaging'),('gradio','gradio'),('diffusers','diffusers'),
        ('transformers','transformers'),('tokenizers','tokenizers'),('peft','peft'),
        ('huggingface_hub','huggingface-hub'),('trl','trl'),('bitsandbytes','bitsandbytes')]:
        module=sys.modules.get(module_name)
        observed[package_name]=None if module is None else {'version':str(getattr(module,'__version__','unreported')),
            'file':str(getattr(module,'__file__','unreported'))}
    return {'pid':os.getpid(),'python_executable':sys.executable,'python':platform.python_version(),
        'ipython_version':IPython.__version__,'jedi_version':jedi.__version__,
        'jedi_completion_passed':'upper' in names,'already_loaded_modules':observed}


def verify_saved_repair(root,helper):
    folder=root/'resume_base_jedi_attempts'/REPAIR_ATTEMPT
    previous=read(folder/'REPORT.json')
    require(previous['status']=='RESUMED_BASE_JEDI_REPAIRED_BOTH_DEPENDENCY_CHECKS_PASSED_RESTART_PENDING' and
        previous['source_sha256']==REPAIR_SOURCE_SHA and previous['base_full_dependency_check_passed'] and
        previous['isolated_full_dependency_check_passed'] and previous['fresh_notebook_tools_and_jedi_completion_passed'] and
        previous['base_package_names']==724 and previous['base_duplicate_groups']==16 and
        previous['isolated_exact_packages_preserved']==84 and previous['base_kernel_restart_required'] is True and
        previous['gpu_exceptions_accepted']==[] and previous['isolated_package_changes_performed'] is False and
        previous['historical_contracts_rewritten'] is False and
        previous['pretrained_model_loads']==previous['generation_calls']==previous['judge_calls']==0,
        'Successful Jedi-only repair evidence differs from the returned result')
    for index,(relative,digest) in enumerate(previous['artifact_sha256'].items(),1):
        require(sha(helper.safe_file(folder,relative))==digest,'Saved repair artifact differs: '+relative)
        if index%10==0:print('  Saved repair artifacts verified:',index,'/',len(previous['artifact_sha256']),flush=True)
    proof_path=folder/'BASE_REPAIR_VALIDATION_CONTRACT.json'
    require(sha(proof_path)==previous['base_repair_validation_contract_sha256'],'Fresh base repair contract differs')
    proof=read(proof_path)
    require(proof['source_sha256']==REPAIR_SOURCE_SHA and proof['only_approved_base_addition']=={'jedi':'0.20.0'} and
        proof['base_full_dependency_check_passed'] and proof['isolated_full_dependency_check_passed'] and
        proof['fresh_notebook_tool_imports_and_jedi_completion_passed'] and
        proof['unrelated_base_metadata_and_search_paths_preserved'],'Saved Jedi repair validation is incomplete')
    require(sha(folder/'JEDI_REPAIR_SOURCE.py')==REPAIR_SOURCE_SHA,'Saved executing repair source differs')
    return folder,previous,proof_path,proof


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
    helper_path=ROOT/'resume_preflight_attempts/20261012T040418.708090Z/RESUME_CHECK_SOURCE.py'
    require(sha(helper_path)==RESUME_SOURCE_SHA,'Saved resume helper differs')
    helper=load('task4_post_resume_helpers',helper_path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt=ROOT/'resume_post_restart_attempts'/stamp;attempt.mkdir(parents=True,exist_ok=False)
    report={'format':'ATML_TASK4_RESUMED_POST_RESTART_V1','attempt_utc':stamp,'status':'RUNNING',
        'source_sha256':sha(__file__),'artifact_sha256':{},'commands':[],
        'package_changes_performed':False,'downloads':0,'environment_created':False,'automatic_restart_performed':False,
        'pretrained_model_loads':0,'generation_calls':0,'judge_calls':0,'training_performed':False,
        'historical_contracts_rewritten':False,'gpu_exceptions_accepted':[],'judge_model_loading_validation_pending':True}
    def save(name,value):
        path=attempt/name;path.parent.mkdir(parents=True,exist_ok=True)
        payload=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        require(not path.exists() and not path.is_symlink(),'Refusing evidence overwrite')
        temporary=path.parent/('.pending-'+path.name)
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        require(not path.exists(),'Concurrent writer created evidence');os.replace(temporary,path)
        require(path.read_bytes()==payload,'Evidence readback failed')
        if name!='REPORT.json':report['artifact_sha256'][name]=sha(path)
    def command(args,label,environment):
        args=[str(x) for x in args];event={'stage':label,'argv':args};save(label+'_COMMAND.json',event)
        print('Post-restart validation:',label,flush=True)
        log_path=attempt/(label+'.log')
        with log_path.open('x') as log:
            process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment,cwd='/content')
            try:
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            log.flush();os.fsync(log.fileno())
        event.update(returncode=code,log_sha256=sha(log_path));report['commands'].append(event)
        save(label+'_RESULT.json',event);require(code==0,label+' failed; no automatic fallback or repair')
    try:
        save('POST_RESTART_SOURCE.py',Path(__file__).read_bytes())
        print('Step 1/5: verifying successful Jedi repair evidence and the original sealed helpers.',flush=True)
        repair_folder,previous,base_path,base=verify_saved_repair(ROOT,helper)
        report['base_repair_report_sha256']=sha(repair_folder/'REPORT.json')
        report['base_repair_validation_contract_sha256']=sha(base_path)
        restore_folder=ROOT/'resume_isolated_restore_attempts/20261012T043404.672313Z'
        restore_proof_path=restore_folder/'ISOLATED_VALIDATION_CONTRACT.json'
        require(sha(restore_folder/'REPORT.json')==base['isolated_restore_report_sha256'] and
            sha(restore_proof_path)==base['isolated_validation_contract_sha256'],'Bound restored-isolation evidence differs')
        isolated=read(restore_proof_path)
        require(isolated['source_sha256']==RESTORE_SOURCE_SHA and isolated['approval_sha256']==base['approval_sha256'] and
            isolated['gpu_exceptions_accepted']==[],'Paired current-runtime approval differs')
        require(LOCAL.is_dir() and not LOCAL.is_symlink() and sha(LOCAL/'.task4_isolation_owner.json')==base['local_owner_sha256']==isolated['local_owner_sha256'],
            'Owned local environment missing or changed. Retain this A100 and local files.')
        owner=read(LOCAL/'.task4_isolation_owner.json')
        require(owner['source_sha256']==RESTORE_SOURCE_SHA and owner['approval_sha256']==base['approval_sha256'] and
            owner['wheelhouse_contract_sha256']==base['wheelhouse_contract_sha256'] and owner['system_site_packages'] is False,
            'Restored environment ownership differs')
        kinds=['preflight','post_restart','isolated_gpu','isolated_recheck','base_repair_recheck',
            'base_layout_diagnostic','gpu_environment','orchestration','diagnostic']
        folders={}
        for index,kind in enumerate(kinds,1):
            print(f'  Source integrity {index}/{len(kinds)}: {kind}',flush=True)
            identity=helper.SOURCES[kind];folder=ROOT/(kind+'_releases')/identity
            require(not folder.is_symlink() and sha(folder/'MANIFEST.json')==identity,'Sealed helper identity differs')
            for relative,digest in read(folder/'MANIFEST.json').items():
                require(sha(helper.safe_file(folder,relative))==digest,'Sealed helper payload differs: '+kind+'/'+relative)
            folders[kind]=folder
        kernel_guards=helper.load('task4_post_resume_kernel_guards',folders['post_restart']/'kernel_guards.py')
        guards=helper.load('task4_post_resume_isolation_guards',folders['isolated_gpu']/'guards.py')
        policy=helper.load('task4_post_resume_child_policy',folders['isolated_recheck']/'isolation.py')
        gates=helper.load('task4_post_resume_hardware_gates',folders['gpu_environment']/'gates.py')
        require(sha(folders['post_restart']/'APPROVAL.json')==base['approval_sha256'] and
            read(folders['post_restart']/'APPROVAL.json')['approved'] is True,'Existing restart/recheck approval differs')
        print('Step 2/5: checking the actual restarted notebook PID, loaded versions and Jedi completion.',flush=True)
        require(os.getpid()!=base['pre_restart_notebook_pid'],
            'Notebook has not restarted. Use ordinary Restart session/kernel, keeping local A100 files, then rerun this block.')
        kernel=observe_kernel();save('RESTARTED_NOTEBOOK_KERNEL.json',kernel)
        previous_kernel=read(repair_folder/'NOTEBOOK_KERNEL_OBSERVATION.json')
        kernel_result=kernel_guards.validate_kernel(kernel,previous_kernel,kernel_guard_view(base),owner)
        save('KERNEL_RESTART_VALIDATION.json',kernel_result)
        print('Actual notebook restart and repaired tools/completion: PASS.',flush=True)
        base_env=guards.clean_environment(os.environ);child_env=policy.child_environment(os.environ,guards.clean_environment)
        parent_backend=os.environ.get('MPLBACKEND')
        gpu=helper.command(['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits'])
        save('CURRENT_GPU_QUERY.json',gpu)
        require(gpu.get('returncode')==0 and gates.parse_a100_inventory(gpu['stdout'])==base['hardware'],
            'GPU/runtime changed after kernel restart; current A100 recheck required')
        print('Step 3/5: checking the exact repaired base layout, full dependencies and fresh notebook tools.',flush=True)
        base_before=helper.inventory(sys.executable);save('BASE_BEFORE_INVENTORY.json',base_before)
        require(base_before==read(repair_folder/'BASE_AFTER_INVENTORY.json'),'Base inventory/search paths changed after repair')
        scanner=folders['base_repair_recheck']/'scan_base_layout.py';raw_scanner=folders['base_layout_diagnostic']/'scan_layout.py'
        command([sys.executable,'-I','-B',scanner,raw_scanner,attempt/'BASE_LAYOUT.json'],'base_layout',base_env)
        require(read(attempt/'BASE_LAYOUT.json')==read(repair_folder/'BASE_AFTER_LAYOUT.json'),
            'Base distribution metadata, duplicates or named selection changed after repair')
        command([sys.executable,'-I','-B','-m','pip','--disable-pip-version-check','check'],'base_full_dependency_check',base_env)
        command([sys.executable,'-I','-B','-u',folders['base_repair_recheck']/'probe_base.py',repair_folder/'EXPECTED_BASE_TOOLS.json',
            attempt/'BASE_TOOLS_PROBE.json',attempt/'BASE_LAYOUT.json'],'preserved_fresh_base_tools_probe',base_env)
        print('Step 4/5: checking the unchanged isolated pins, full dependencies and fresh CUDA/NF4/37-check probe.',flush=True)
        pins=isolated['isolated_package_versions']
        require(len(pins)==84 and pins==read(folders['isolated_gpu']/'APPROVED_ISOLATED_VERSIONS.json'),'Exact approved isolated pins differ')
        python=LOCAL/'bin/python';inspect=folders['isolated_gpu']/'inspect_python.py'
        command([python,'-I','-B',inspect,attempt/'ISOLATED_BEFORE.json'],'isolated_inventory_before',child_env)
        iso_before=read(attempt/'ISOLATED_BEFORE.json');guards.validate_inventory(iso_before,pins,LOCAL,complete=True)
        require(iso_before==read(repair_folder/'ISOLATED_AFTER.json'),'Isolated interpreter differs after restart')
        command([python,'-I','-B','-m','pip','--disable-pip-version-check','check'],'isolated_full_dependency_check',child_env)
        probe_dir=attempt/'fresh_probe';probe_dir.mkdir()
        command([python,'-I','-B','-u',folders['gpu_environment']/'probe_gpu.py',folders['preflight'],
            folders['orchestration'],folders['diagnostic'],probe_dir],'preserved_fresh_a100_probe',child_env)
        probe=read(probe_dir/'GPU_PROBE.json')
        require(probe['all_required_package_versions']=={name:version for name,version in pins.items() if name!='pip'} and
            probe['required_distributions']==83 and probe['synthetic_semantic_checks_passed']==37 and
            probe['required_dependency_chain_passed'] and probe['fresh_process_imports_passed'] and
            probe['tiny_float16_cuda_check_passed'] and probe['tiny_nf4_cuda_backend_check_passed'] and
            probe['pretrained_model_weights_loaded'] is False,'Post-restart isolated validation incomplete')
        print('Step 5/5: confirming unchanged inventories and saving fresh runtime evidence separately.',flush=True)
        base_after=helper.inventory(sys.executable);save('BASE_AFTER_INVENTORY.json',base_after)
        command([python,'-I','-B',inspect,attempt/'ISOLATED_AFTER.json'],'isolated_inventory_after',child_env)
        iso_after=read(attempt/'ISOLATED_AFTER.json');guards.validate_inventory(iso_after,pins,LOCAL,complete=True)
        require(base_after==base_before and iso_before==iso_after and os.environ.get('MPLBACKEND')==parent_backend,
            'Current base/isolated inventories or parent backend changed during recheck')
        require(sha(LOCAL/'.task4_isolation_owner.json')==base['local_owner_sha256'] and
            sha(base_path)==report['base_repair_validation_contract_sha256'],'Fresh source evidence changed during recheck')
        for relative,digest in base['historical_metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT,relative))==digest,'Historical generation/preparation contract changed: '+relative)
        current_contract={'format':'ATML_TASK4_RESUMED_CURRENT_RUNTIME_V1','source_sha256':report['source_sha256'],
            'approval_sha256':base['approval_sha256'],'base_repair_report_sha256':report['base_repair_report_sha256'],
            'base_repair_validation_contract_sha256':sha(base_path),'isolated_validation_contract_sha256':sha(restore_proof_path),
            'wheelhouse_contract_sha256':base['wheelhouse_contract_sha256'],'local_owner_sha256':base['local_owner_sha256'],
            'hardware':base['hardware'],'base_python_executable':sys.executable,'isolated_python_executable':str(python),
            'python':iso_after['python'],'base_inventory_sha256':sha(attempt/'BASE_AFTER_INVENTORY.json'),
            'base_layout_sha256':sha(attempt/'BASE_LAYOUT.json'),'isolated_inventory_sha256':sha(attempt/'ISOLATED_AFTER.json'),
            'isolated_package_versions':pins,'base_selected_metadata':base_after['inventory']['selected_packages'],
            'base_duplicate_groups':base_after['inventory']['duplicate_groups'],'notebook_pid':kernel['pid'],
            'kernel_restart_verified':True,'actual_notebook_tools_and_jedi_completion_passed':True,
            'base_full_dependency_check_passed':True,'isolated_full_dependency_check_passed':True,
            'fresh_imports_cuda_nf4_passed':True,'synthetic_checks_passed':37,'gpu_exceptions_accepted':[],
            'historical_metadata_sha256':base['historical_metadata_sha256'],
            'sealed_sources_used':{kind:helper.SOURCES[kind] for kind in kinds},
            'runtime_ready_for_judge_loader_validation':True,'judge_model_loading_validation_pending':True,
            'generation_complete':True,'judging_execution_pending':True}
        save('CURRENT_RUNTIME_VALIDATION_CONTRACT.json',current_contract)
        report.update(status='RESUMED_BOTH_INTERPRETERS_VALIDATED_AFTER_RESTART_NO_INFERENCE',
            kernel_restart_verified=True,base_package_names=len(base_after['inventory']['selected_packages']),
            base_duplicate_groups=len(base_after['inventory']['duplicate_groups']),isolated_package_count=84,
            base_full_dependency_check_passed=True,isolated_full_dependency_check_passed=True,
            fresh_imports_cuda_nf4_passed=True,synthetic_checks_passed=37,
            current_runtime_validation_contract_sha256=sha(attempt/'CURRENT_RUNTIME_VALIDATION_CONTRACT.json'))
    except BaseException as error:
        import traceback
        report['status']='RESUMED_POST_RESTART_CHECK_STOPPED_EVIDENCE_PRESERVED'
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name!='REPORT.json':
            report['artifact_sha256'][str(path.relative_to(attempt))]=sha(path)
    save('REPORT.json',report)
    print('Saved report:',attempt/'REPORT.json',flush=True)
    print('Status:',report['status'],flush=True)
    if report['status']=='RESUMED_BOTH_INTERPRETERS_VALIDATED_AFTER_RESTART_NO_INFERENCE':
        print('Actual kernel restart PASS; both full dependency checks PASS; fresh imports/CUDA/NF4 PASS; 37/37 checks PASS.',flush=True)
        print('Base:',report['base_package_names'],'names;',report['base_duplicate_groups'],'preserved duplicate groups; isolated: 84 exact packages.',flush=True)
        print('Fresh runtime evidence saved separately. Original judge loading validation remains pending.',flush=True)
    else:print(json.dumps(report.get('error'),indent=2),flush=True)
    print('Responses retained: 1800/1800; judgments: 0/1800; human labels last confirmed: 0/240.',flush=True)
    print('No installations, downloads, model loading, generation, judging or training. Keep this A100 and return the full output.',flush=True)


if __name__=='__main__':main()
