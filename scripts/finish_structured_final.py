"""One-shot handoff from the existing training process to final assessment.

Never starts training, retries a failed stage, or overwrites an evaluation.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.train_compact_residual import digest,write_json


def execute_stages(stages,out,run=subprocess.run):
    completed=[]
    for name,args in stages:
        write_json(out/'progress.json',dict(stage=name,completed=completed,status='running'))
        print(f'Starting {name}; log: {out/name}.log',flush=True)
        with (out/f'{name}.log').open('w') as log:
            run([sys.executable,'-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        completed.append(name)
    write_json(out/'progress.json',dict(stage='complete',completed=completed,status='complete'))


def wait_existing_process(pid,deadline):
    """Hold an OS handle so PID reuse cannot switch the process being watched."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000|0x1000,False,pid)
    if not handle:raise OSError(ctypes.get_last_error(),'cannot attach to the existing training process')
    try:
        while True:
            status=kernel.WaitForSingleObject(handle,1000)
            if status==0:
                code=wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError('cannot read training exit code')
                if code.value:raise RuntimeError(f'training process exited {code.value}; no next stage')
                return
            if status!=258:raise OSError('training process wait failed')
            if time.monotonic()>deadline:raise TimeoutError('handoff wait budget; training is not terminated')
    finally:kernel.CloseHandle(handle)


def run(pid,output,evaluation):
    out=Path(output);evaluation=Path(evaluation)
    if evaluation.exists():raise FileExistsError('existing evaluation must never be repeated')
    out.mkdir(parents=True,exist_ok=False)
    training=Path('outputs/structured_final/training_recovered_v1')
    sources=list(Path('src').rglob('*.py'))+list(Path('scripts').glob('*.py'))
    sources += [Path(p) for p in ['STRUCTURED_FINAL_STUDY_PROTOCOL.md','STRUCTURED_TRAINING_RECOVERY.md','Docs/3. Structured RL study adoption.md']]
    hashes={str(p):digest(p) for p in sources}
    write_json(out/'manifest.json',dict(training_pid=pid,training=str(training),evaluation=str(evaluation),
        hashes=hashes,retries=0,training_launches=0))
    stage='waiting_for_existing_training'
    try:
        write_json(out/'progress.json',dict(stage=stage,status='waiting',training_pid=pid))
        print(f'Attached handoff to existing training PID {pid}; no new training launched.',flush=True)
        wait_existing_process(pid,time.monotonic()+30000)
        if (training/'failed_status.json').exists():raise RuntimeError('training failure marker; no next stage')
        summary=json.loads((training/'summary.json').read_text())
        if summary['status']!='complete' or len(summary['runs'])!=25:raise ValueError('complete25-run cohort required')
        for path,h in hashes.items():
            if digest(path)!=h:raise ValueError(f'handoff source changed: {path}')
        stage='verification_and_assessment'
        execute_stages([
            ('training_verification',['scripts.verify_recovered_structured_training']),
            ('evaluation',['scripts.evaluate_structured_final','--training',str(training),'--output',str(evaluation)]),
            ('evaluation_verification',['scripts.verify_structured_final','--output',str(evaluation)]),
            ('report',['scripts.report_structured_final','--output',str(evaluation)]),
        ],out)
        print('Final study artifacts generated; visual QA and report/demo integration remain.',flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),retries=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training-pid',required=True,type=int)
    p.add_argument('--output',required=True);p.add_argument('--evaluation',required=True)
    a=p.parse_args();run(a.training_pid,a.output,a.evaluation)
