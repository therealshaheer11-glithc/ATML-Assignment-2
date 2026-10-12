"""Install only the already-approved Jedi 0.20.0 in the current Colab base."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL=Path('/content/task4_isolated_runtime_v1')
RESTORE_ATTEMPT='20261012T043404.672313Z'
RESTORE_SOURCE_SHA='a3025083ed0094652e39e306c7d2bdaf226d025a651e705cf3f8f37b17ac130e'
RESUME_SOURCE_SHA='818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'


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


def validate_jedi_only_inventory(current,baseline):
    require('jedi' not in baseline['selected_packages'],'Saved pre-repair base unexpectedly contains Jedi')
    candidate=deepcopy(current);entry=candidate['selected_packages'].pop('jedi',None)
    require(candidate==baseline,'Base inventory/search paths/duplicates changed beyond the approved Jedi addition')
    if entry is not None:
        require(entry['version']=='0.20.0' and Path(entry['root']).resolve().is_relative_to(Path(current['prefix']).resolve()),
                'Existing Jedi differs from the approved exact version/location')
    require(current['prefix']==current['base_prefix'] and current['user_site_enabled'] is False and
            not Path(current['prefix']).is_relative_to(LOCAL),'Repair interpreter is not the approved Colab base')
    return entry is not None


def validate_jedi_only_layout(before,after):
    left,right=deepcopy(before),deepcopy(after)
    require('jedi' not in before['duplicate_distribution_entries'] and 'jedi' not in after['duplicate_distribution_entries'],
            'Jedi has ambiguous duplicate metadata')
    for field in ['package_versions','all_distribution_entries','selected_distribution_metadata','effective_metadata_versions']:
        left[field].pop('jedi',None);right[field].pop('jedi',None)
    require(left==right,'An unrelated base metadata entry, selected version, duplicate group or search path changed')
    entries=after['all_distribution_entries'].get('jedi',[])
    require(len(entries)==1 and entries[0]['version']=='0.20.0' and
            after['selected_distribution_metadata']['jedi']==entries[0] and
            after['effective_metadata_versions']['jedi']==after['package_versions']['jedi']=='0.20.0',
            'Final Jedi metadata/version is not the exact approved singleton')


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
    previous_folder=ROOT/'resume_isolated_restore_attempts'/RESTORE_ATTEMPT
    restore_source=previous_folder/'RESTORE_SOURCE.py'
    require(sha(restore_source)==RESTORE_SOURCE_SHA,'Saved restore source differs')
    restore=load('task4_jedi_restore_helpers',restore_source)
    helper_path=ROOT/'resume_preflight_attempts/20261012T040418.708090Z/RESUME_CHECK_SOURCE.py'
    require(sha(helper_path)==RESUME_SOURCE_SHA,'Saved resume helper differs')
    helper=load('task4_jedi_resume_helpers',helper_path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt=ROOT/'resume_base_jedi_attempts'/stamp;attempt.mkdir(parents=True,exist_ok=False)
    report={'format':'ATML_TASK4_RESUMED_BASE_JEDI_REPAIR_V1','attempt_utc':stamp,'status':'RUNNING',
        'source_sha256':sha(__file__),'artifact_sha256':{},'commands':[],
        'scope':'Already-approved Jedi 0.20.0-only base repair after successful isolated validation.',
        'base_installation_attempted':False,'isolated_package_changes_performed':False,'historical_contracts_rewritten':False,
        'downloads':0,'pretrained_model_loads':0,'generation_calls':0,'judge_calls':0,'training_performed':False,
        'gpu_exceptions_accepted':[],'kernel_restart_performed':False,'base_kernel_restart_required':True}
    def save(name,value):
        path=attempt/name;path.parent.mkdir(parents=True,exist_ok=True)
        payload=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        require(not path.exists() and not path.is_symlink(),'Refusing evidence overwrite')
        temporary=path.parent/('.pending-'+path.name)
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        require(not path.exists(),'Concurrent writer created evidence');os.replace(temporary,path)
        require(path.read_bytes()==payload,'Saved evidence readback failed')
        if name!='REPORT.json':report['artifact_sha256'][name]=sha(path)
    def command(args,label,environment):
        args=[str(x) for x in args];event={'stage':label,'argv':args}
        save(label+'_COMMAND.json',event);print('Runtime repair:',label,flush=True)
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
        save(label+'_RESULT.json',event)
        require(code==0,label+' failed; evidence preserved without automatic repair or exception')
    try:
        save('JEDI_REPAIR_SOURCE.py',Path(__file__).read_bytes())
        print('Step 1/6: verifying the successful restored-environment report and saved validation evidence.',flush=True)
        previous=read(previous_folder/'REPORT.json')
        require(previous['status']=='RESUMED_ISOLATED_A100_VALIDATED_BASE_REPAIR_PENDING' and
                previous['source_sha256']==RESTORE_SOURCE_SHA and previous['fresh_wheels_verified']==90 and
                previous['isolated_distribution_count']==84 and previous['required_distribution_count']==83 and
                previous['synthetic_checks_passed']==37 and previous['isolated_full_dependency_check_passed'] and
                previous['fresh_imports_cuda_nf4_passed'] and previous['base_package_inventory_and_layout_unchanged'] and
                previous['base_package_changes_performed'] is False and previous['gpu_exceptions_accepted']==[],
                'Returned isolated restoration success differs')
        for index,(relative,digest) in enumerate(previous['artifact_sha256'].items(),1):
            require(sha(helper.safe_file(previous_folder,relative))==digest,'Restoration artifact differs: '+relative)
            if index%20==0:print('  Saved restoration artifacts verified:',index,'/',len(previous['artifact_sha256']),flush=True)
        proof_path=previous_folder/'ISOLATED_VALIDATION_CONTRACT.json'
        require(sha(proof_path)==previous['isolated_validation_contract_sha256'],'Fresh isolated validation contract differs')
        proof=read(proof_path);report['isolated_restore_report_sha256']=sha(previous_folder/'REPORT.json')
        report['isolated_validation_contract_sha256']=sha(proof_path)
        require(proof['source_sha256']==RESTORE_SOURCE_SHA and proof['gpu_exceptions_accepted']==[] and
                proof['fresh_imports_cuda_nf4_passed'] and proof['synthetic_checks_passed']==37,
                'Fresh isolated proof is incomplete')
        require(sha(LOCAL/'.task4_isolation_owner.json')==proof['local_owner_sha256'] and
                read(LOCAL/'.task4_isolation_owner.json')==read(previous_folder/'LOCAL_OWNER.json'),'Restored local owner differs')
        owner=read(LOCAL/'.task4_isolation_owner.json')
        restore.check_existing_owner(LOCAL,owner)
        require(owner['source_sha256']==RESTORE_SOURCE_SHA and owner['approval_sha256']==proof['approval_sha256'] and
                owner['wheelhouse_contract_sha256']==proof['wheelhouse_contract_sha256'] and
                owner['target_python_full_version']==proof['python'] and owner['system_site_packages'] is False and
                str(Path(sys.executable).resolve())==owner['base_python_executable'],'Owned current interpreter differs')
        folders={}
        used=['isolated_gpu','isolated_recheck','base_repair_recheck','base_layout_diagnostic','wheel_cache','diagnostic','gpu_environment']
        for index,kind in enumerate(used,1):
            print(f'  Source integrity {index}/{len(used)}: {kind}',flush=True)
            identity=helper.SOURCES[kind];folder=ROOT/(kind+'_releases')/identity
            require(not folder.is_symlink() and sha(folder/'MANIFEST.json')==identity,'Sealed helper source differs')
            for relative,digest in read(folder/'MANIFEST.json').items():
                require(sha(helper.safe_file(folder,relative))==digest,'Sealed helper payload differs: '+kind+'/'+relative)
            folders[kind]=folder
        guards=helper.load('task4_jedi_isolation_guards',folders['isolated_gpu']/'guards.py')
        isolation=helper.load('task4_jedi_child_policy',folders['isolated_recheck']/'isolation.py')
        repair=helper.load('task4_jedi_original_repair_commands',folders['base_repair_recheck']/'repair_guards.py')
        wheels=helper.load('task4_jedi_wheel_guards',folders['wheel_cache']/'wheels.py')
        dependencies=helper.load('task4_jedi_dependencies',folders['diagnostic']/'dependencies.py')
        gates=helper.load('task4_jedi_hardware_gates',folders['gpu_environment']/'gates.py')
        base_env=guards.clean_environment(os.environ);child_env=isolation.child_environment(os.environ,guards.clean_environment)
        parent_backend=os.environ.get('MPLBACKEND')
        gpu=helper.command(['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits'])
        save('CURRENT_GPU_QUERY.json',gpu)
        require(gpu.get('returncode')==0 and gates.parse_a100_inventory(gpu['stdout'])==proof['hardware'],
                'A100 changed since successful restoration')
        for relative,digest in proof['historical_metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT,relative))==digest,'Historical preparation contract differs: '+relative)
        approval=read(folders['isolated_gpu']/'APPROVAL.json')
        require(approval['approved'] is True and approval['gpu_exceptions_authorized'] is False and
                'conditional original Colab base restoration and Jedi repair after isolated validation' in approval['approved_scope'] and
                sha(folders['isolated_gpu']/'APPROVAL.json')==proof['approval_sha256'],
                'Existing exact Jedi repair authorization differs')
        print('Step 2/6: checking both current inventories and freezing the actual base metadata layout.',flush=True)
        baseline=read(previous_folder/'BASE_AFTER_INVENTORY.json')['inventory']
        before=helper.inventory(sys.executable);save('BASE_BEFORE_INVENTORY.json',before)
        require('inventory' in before,'Base inventory failed')
        already_installed=validate_jedi_only_inventory(before['inventory'],baseline)
        selected=before['inventory']['selected_packages']
        expected_tools={'ipython':'7.34.0','jedi':'0.20.0','parso':'0.8.7','gradio':'6.29.0','diffusers':'0.40.0'}
        for name,version in expected_tools.items():
            if name!='jedi':require(selected.get(name,{}).get('version')==version,'Notebook tool or Parso version changed: '+name)
        python=LOCAL/'bin/python';inspect=folders['isolated_gpu']/'inspect_python.py'
        command([python,'-I','-B',inspect,attempt/'ISOLATED_BEFORE.json'],'isolated_inventory_before',child_env)
        isolated_before=read(attempt/'ISOLATED_BEFORE.json')
        guards.validate_inventory(isolated_before,proof['isolated_package_versions'],LOCAL,complete=True)
        require(isolated_before==read(previous_folder/'ISOLATED_AFTER.json'),'Isolated package/path inventory changed after restoration')
        scanner=folders['base_repair_recheck']/'scan_base_layout.py';raw_scanner=folders['base_layout_diagnostic']/'scan_layout.py'
        command([sys.executable,'-I','-B',scanner,raw_scanner,attempt/'BASE_BEFORE_LAYOUT.json'],'base_layout_before',base_env)
        layout_before=read(attempt/'BASE_BEFORE_LAYOUT.json')
        require(layout_before['prefix']==layout_before['base_prefix']==before['inventory']['prefix'] and
                layout_before['user_site_enabled'] is False and not layout_before['outside_prefix_distribution_entries'] and
                not layout_before['outside_prefix_site_paths'] and 'jedi' not in layout_before['duplicate_distribution_entries'],
                'Base metadata path/isolation or Jedi ambiguity requires review')
        require(layout_before['effective_metadata_versions']=={name:entry['version'] for name,entry in selected.items()},
                'Detailed layout and actual selected metadata disagree')
        save('NOTEBOOK_KERNEL_OBSERVATION.json',{'pid':os.getpid(),'python_executable':sys.executable,
            'python':before['inventory']['python'],'ipython_version':selected['ipython']['version'],
            'parent_mplbackend':parent_backend})
        print('Step 3/6: freshly verifying only the approved Jedi wheel and its existing Parso dependency.',flush=True)
        wheelhouse=ROOT/'environment_wheelhouse';wheel_path=wheelhouse/'VERIFIED_WHEELHOUSE_CONTRACT.json'
        require(sha(wheel_path)==proof['wheelhouse_contract_sha256'],'Approved wheelhouse changed')
        cache=read(wheel_path)
        from packaging.markers import default_environment
        from packaging.tags import sys_tags
        from packaging.utils import canonicalize_name
        matches=[record for record in cache['files'].values() if canonicalize_name(record['package'])=='jedi' and record['version']=='0.20.0']
        require(len(matches)==1,'Exact approved Jedi wheel missing or duplicated')
        record=matches[0];wheel=wheelhouse/'wheels'/record['filename']
        result=wheels.verify_file(wheel,record)
        require(result==cache['verified_files'][record['filename']]==read(wheelhouse/'receipts'/(record['filename']+'.json')),
                'Jedi wheel receipt differs')
        inspection=wheels.inspect_wheel(wheel,record,list(sys_tags()),default_environment()['python_full_version'])
        require(inspection==cache['wheel_inspections'][record['filename']],'Jedi wheel metadata differs')
        import importlib.metadata as metadata
        def lookup(name):
            if canonicalize_name(name)=='jedi':return inspection['metadata']
            try:distribution=metadata.distribution(name)
            except metadata.PackageNotFoundError:return None
            return {'name':distribution.metadata['Name'],'version':distribution.version,
                'requires_dist':distribution.requires or [],'provides_extra':distribution.metadata.get_all('Provides-Extra',[]),
                'requires_python':distribution.metadata.get('Requires-Python')}
        graph=dependencies.inspect_chain(['jedi==0.20.0'],lookup,default_environment())
        require(graph['required_chain_passed'] and not graph['issues'] and set(graph['nodes'])=={'jedi','parso'} and
                graph['nodes']['parso']['version']=='0.8.7','Jedi cannot use the unchanged installed Parso')
        save('APPROVED_JEDI_DEPENDENCY_CHAIN.json',graph)
        save('OFFLINE_JEDI_REQUIREMENT.txt',guards.requirement_lines(cache['files'],{'jedi':'0.20.0'},wheelhouse/'wheels').encode())
        save('EXPECTED_BASE_TOOLS.json',expected_tools)
        print('Step 4/6: installing only Jedi 0.20.0 offline; all other base packages remain fixed.',flush=True)
        if not already_installed:
            report['base_installation_attempted']=True
            command(repair.base_install_command(sys.executable,attempt/'OFFLINE_JEDI_REQUIREMENT.txt',attempt/'BASE_INSTALL_REPORT.json'),
                'install_exact_approved_jedi',base_env)
        else:print('Exact approved Jedi already present; no repeat installation.',flush=True)
        print('Step 5/6: complete base dependency check, metadata preservation, fresh tools and completion.',flush=True)
        after=helper.inventory(sys.executable);save('BASE_AFTER_INVENTORY.json',after)
        require('inventory' in after and validate_jedi_only_inventory(after['inventory'],baseline),
                'Post-repair base does not contain only the exact approved Jedi addition')
        command([sys.executable,'-I','-B',scanner,raw_scanner,attempt/'BASE_AFTER_LAYOUT.json'],'base_layout_after',base_env)
        layout_after=read(attempt/'BASE_AFTER_LAYOUT.json');validate_jedi_only_layout(layout_before,layout_after)
        command([sys.executable,'-I','-B','-m','pip','--disable-pip-version-check','check'],'base_full_dependency_check',base_env)
        command([sys.executable,'-I','-B','-u',folders['base_repair_recheck']/'probe_base.py',attempt/'EXPECTED_BASE_TOOLS.json',
            attempt/'BASE_TOOLS_PROBE.json',attempt/'BASE_AFTER_LAYOUT.json'],'preserved_fresh_base_tools_probe',base_env)
        probe=read(attempt/'BASE_TOOLS_PROBE.json')
        require(probe['jedi_string_upper_completion_passed'] and set(probe['fresh_tool_imports'])==set(expected_tools),
                'Fresh notebook tool validation is incomplete')
        print('Step 6/6: confirming the isolated 84-package environment and historical generation evidence are unchanged.',flush=True)
        command([python,'-I','-B',inspect,attempt/'ISOLATED_AFTER.json'],'isolated_inventory_after',child_env)
        isolated_after=read(attempt/'ISOLATED_AFTER.json');guards.validate_inventory(isolated_after,proof['isolated_package_versions'],LOCAL,complete=True)
        require(isolated_after==isolated_before,'Base repair changed the isolated interpreter')
        command([python,'-I','-B','-m','pip','--disable-pip-version-check','check'],'isolated_full_dependency_check',child_env)
        require(os.environ.get('MPLBACKEND')==parent_backend and sha(LOCAL/'.task4_isolation_owner.json')==proof['local_owner_sha256'],
                'Parent backend or local owner changed')
        for relative,digest in proof['historical_metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT,relative))==digest,'Historical generation/preparation contract changed: '+relative)
        contract={'format':'ATML_TASK4_RESUMED_BASE_REPAIR_VALIDATION_V1','source_sha256':report['source_sha256'],
            'approval_sha256':proof['approval_sha256'],'isolated_restore_report_sha256':report['isolated_restore_report_sha256'],
            'isolated_validation_contract_sha256':sha(proof_path),'local_owner_sha256':proof['local_owner_sha256'],
            'wheelhouse_contract_sha256':proof['wheelhouse_contract_sha256'],'hardware':proof['hardware'],
            'base_python_executable':sys.executable,'python':after['inventory']['python'],
            'base_selected_metadata':after['inventory']['selected_packages'],'base_duplicate_groups':after['inventory']['duplicate_groups'],
            'base_inventory_sha256':sha(attempt/'BASE_AFTER_INVENTORY.json'),'base_layout_sha256':sha(attempt/'BASE_AFTER_LAYOUT.json'),
            'isolated_inventory_sha256':sha(attempt/'ISOLATED_AFTER.json'),'isolated_exact_packages':84,
            'base_full_dependency_check_passed':True,'isolated_full_dependency_check_passed':True,
            'fresh_notebook_tool_imports_and_jedi_completion_passed':True,'unrelated_base_metadata_and_search_paths_preserved':True,
            'only_approved_base_addition':{'jedi':'0.20.0'},'historical_metadata_sha256':proof['historical_metadata_sha256'],
            'base_kernel_restart_required':True,'pre_restart_notebook_pid':os.getpid(),'gpu_exceptions_accepted':[],
            'judge_model_loading_validation_pending':True,'judging_execution_pending':True}
        save('BASE_REPAIR_VALIDATION_CONTRACT.json',contract)
        report.update(status='RESUMED_BASE_JEDI_REPAIRED_BOTH_DEPENDENCY_CHECKS_PASSED_RESTART_PENDING',
            base_full_dependency_check_passed=True,isolated_full_dependency_check_passed=True,
            base_package_names=len(after['inventory']['selected_packages']),base_duplicate_groups=len(after['inventory']['duplicate_groups']),
            fresh_notebook_tools_and_jedi_completion_passed=True,isolated_exact_packages_preserved=84,
            base_repair_validation_contract_sha256=sha(attempt/'BASE_REPAIR_VALIDATION_CONTRACT.json'))
    except BaseException as error:
        import traceback
        report['status']='RESUMED_BASE_JEDI_REPAIR_STOPPED_EVIDENCE_PRESERVED'
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name!='REPORT.json':
            report['artifact_sha256'][str(path.relative_to(attempt))]=sha(path)
    save('REPORT.json',report)
    print('Saved report:',attempt/'REPORT.json',flush=True)
    print('Status:',report['status'],flush=True)
    if report['status']=='RESUMED_BASE_JEDI_REPAIRED_BOTH_DEPENDENCY_CHECKS_PASSED_RESTART_PENDING':
        print('Only approved base addition: Jedi 0.20.0. Both full dependency checks PASS; fresh tools/completion PASS.',flush=True)
        print('Base metadata:',report['base_package_names'],'names;',report['base_duplicate_groups'],'preserved duplicate groups; isolated packages: 84/84.',flush=True)
        print('Ordinary notebook kernel/session restart is required next; retain this A100 and its local files.',flush=True)
    else:print(json.dumps(report.get('error'),indent=2),flush=True)
    print('Responses retained: 1800/1800; judgments: 0/1800; human labels last confirmed: 0/240.',flush=True)
    print('No model loading, generation, judging, training or automatic restart. Return the complete output before the next block.',flush=True)


if __name__=='__main__':main()
