"""Single-buffer JSON encoding with unchanged durable atomic-write semantics."""
import json
import os
from pathlib import Path


def atomic_json(path,value):
    # Same encoder options/bytes as execution_journal.atomic_json, but use the
    # fast complete-document encoder instead of thousands of Python writes.
    encoded=json.dumps(value,allow_nan=False,sort_keys=True)
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
