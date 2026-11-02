import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from src.experiments.windows_worker import OwnedWorker, api, checked

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows job-object tests')


def test_exit_and_timeout(tmp_path):
    with OwnedWorker([sys.executable, '-c', 'raise SystemExit(7)']) as worker:
        assert worker.wait(time.perf_counter()+10) == 7
    began = time.perf_counter()
    with OwnedWorker([sys.executable, '-c', 'import time; time.sleep(60)']) as worker:
        with pytest.raises(TimeoutError):
            worker.wait(began+.5)
        assert worker.process is None and worker.job is None
    assert time.perf_counter()-began < 10


def test_abrupt_parent_exit_kills_worker(tmp_path):
    pid_file = tmp_path/'pid'
    release = tmp_path/'release'
    code = '''
import os,sys,time
from pathlib import Path
from src.experiments.windows_worker import OwnedWorker
child_code="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
w=OwnedWorker([sys.executable,'-c',child_code,sys.argv[1]+'.child'])
Path(sys.argv[1]).write_text(str(w.pid))
while not Path(sys.argv[2]).exists():time.sleep(.02)
os._exit(23)
'''
    parent = subprocess.Popen([sys.executable, '-c', code, str(pid_file), str(release)],
                              creationflags=subprocess.CREATE_NO_WINDOW)
    k = api()
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    handles = []
    try:
        deadline = time.perf_counter()+10
        child_file = Path(str(pid_file)+'.child')
        while not child_file.exists() and parent.poll() is None and time.perf_counter() < deadline:
            time.sleep(.02)
        assert child_file.exists(), parent.poll()
        for path in (pid_file, child_file):
            handles.append(checked(k.OpenProcess(0x100000, False, int(path.read_text()))))
        release.write_text('exit')
        assert parent.wait(timeout=10) == 23
        for handle in handles:
            assert k.WaitForSingleObject(handle, 10000) == 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        for handle in handles:
            k.CloseHandle(handle)
