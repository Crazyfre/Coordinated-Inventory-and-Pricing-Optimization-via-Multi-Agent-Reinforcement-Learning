"""Windows 10+ owned worker tree, assigned atomically at process creation.

The sole non-inherited job handle belongs to the parent. Parent death closes it
and kills the worker tree. This primitive is not yet a scientific dispatcher.
Reference: https://devblogs.microsoft.com/oldnewthing/20230209-00/?p=107812/
"""
import ctypes as C
from ctypes import wintypes as W
import math
import os
import subprocess
import time


class BasicLimits(C.Structure):
    _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64),
                ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t),
                ('active', W.DWORD), ('affinity', C.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]


class IO(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]


class Limits(C.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IO), ('process_memory', C.c_size_t),
                ('job_memory', C.c_size_t), ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


class Startup(C.Structure):
    _fields_ = [('cb', W.DWORD), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR), ('title', W.LPWSTR),
                ('x', W.DWORD), ('y', W.DWORD), ('width', W.DWORD), ('height', W.DWORD),
                ('xchars', W.DWORD), ('ychars', W.DWORD), ('fill', W.DWORD), ('flags', W.DWORD),
                ('show', W.WORD), ('reserved_size', W.WORD), ('reserved_bytes', C.c_void_p),
                ('stdin', W.HANDLE), ('stdout', W.HANDLE), ('stderr', W.HANDLE)]


class StartupEx(C.Structure):
    _fields_ = [('startup', Startup), ('attributes', C.c_void_p)]


class ProcessInfo(C.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', W.DWORD), ('tid', W.DWORD)]


def api():
    if os.name != 'nt':
        raise RuntimeError('Windows worker requires Windows 10 or later')
    k = C.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
        'InitializeProcThreadAttributeList': ([C.c_void_p, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)], W.BOOL),
        'UpdateProcThreadAttribute': ([C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p], W.BOOL),
        'DeleteProcThreadAttributeList': ([C.c_void_p], None),
        'CreateProcessW': ([W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL, W.DWORD,
                            C.c_void_p, W.LPCWSTR, C.POINTER(StartupEx), C.POINTER(ProcessInfo)], W.BOOL),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
        'GetExitCodeProcess': ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
        'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes = args
        fn.restype = result
    return k


def checked(value):
    if not value:
        raise C.WinError(C.get_last_error())
    return value


class OwnedWorker:
    def __init__(self, argv, cwd=None):
        if not argv or not os.path.isabs(argv[0]):
            raise ValueError('absolute executable required')
        self.k = api()
        self.job = None
        self.process = None
        self.pid = None
        attributes = None
        initialized = False
        try:
            self.job = checked(self.k.CreateJobObjectW(None, None))
            limits = Limits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            checked(self.k.SetInformationJobObject(self.job, 9, C.byref(limits), C.sizeof(limits)))
            size = C.c_size_t()
            self.k.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
            if not size.value:
                raise C.WinError(C.get_last_error())
            attributes = C.create_string_buffer(size.value)
            checked(self.k.InitializeProcThreadAttributeList(attributes, 1, 0, C.byref(size)))
            initialized = True
            jobs = (W.HANDLE*1)(self.job)
            checked(self.k.UpdateProcThreadAttribute(attributes, 0, 0x2000D,
                                                     jobs, C.sizeof(jobs), None, None))
            startup = StartupEx()
            startup.startup.cb = C.sizeof(startup)
            startup.attributes = C.cast(attributes, C.c_void_p)
            info = ProcessInfo()
            command = C.create_unicode_buffer(subprocess.list2cmdline(argv))
            # EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW; inherit no handles.
            checked(self.k.CreateProcessW(argv[0], command, None, None, False, 0x08080000,
                                          None, cwd, C.byref(startup), C.byref(info)))
            self.process, self.pid = info.process, info.pid
            self.k.CloseHandle(info.thread)
            assigned = W.BOOL()
            checked(self.k.IsProcessInJob(self.process, self.job, C.byref(assigned)))
            if not assigned.value:
                raise RuntimeError('worker was not assigned to owned job')
        except BaseException:
            self.close()
            raise
        finally:
            if initialized:
                self.k.DeleteProcThreadAttributeList(attributes)

    def poll(self):
        result = self.k.WaitForSingleObject(self.process, 0)
        if result == 258:
            return None
        if result != 0:
            raise C.WinError(C.get_last_error())
        code = W.DWORD()
        checked(self.k.GetExitCodeProcess(self.process, C.byref(code)))
        return code.value

    def wait(self, deadline):
        if not math.isfinite(deadline):
            raise ValueError('finite absolute monotonic deadline required')
        while True:
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                self.close()
                raise TimeoutError('owned worker exceeded deadline; job terminated')
            result = self.k.WaitForSingleObject(self.process, min(1000, math.ceil(remaining*1000)))
            if result == 0:
                code = W.DWORD()
                checked(self.k.GetExitCodeProcess(self.process, C.byref(code)))
                return code.value
            if result != 258:
                raise C.WinError(C.get_last_error())

    def close(self):
        if self.job:
            checked(self.k.CloseHandle(self.job))
            self.job = None
        if self.process:
            # Keep ownership until Windows confirms this worker has stopped.
            result = self.k.WaitForSingleObject(self.process, 10000)
            if result != 0:
                raise RuntimeError('worker termination could not be confirmed')
            checked(self.k.CloseHandle(self.process))
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
