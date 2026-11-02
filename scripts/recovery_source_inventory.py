"""Prospective launch inventory; refuses automatic rebuilding of cached data.

No simulator construction, calibration fit, download, or campaign launch.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from src.experiments.execution_journal import digest

DATA_PATHS=('src/config/skus.json','data/processed/aligned_skus.csv',
            'data/processed/aligned_skus.key.json')
DESIGN='Docs/1. Research Design and Locked Configuration'
PACKAGES=('numpy','pandas','scipy','torch','gymnasium','statsmodels')


def cache_contract(root):
    root=Path(root)
    for name in DATA_PATHS:
        if not (root/name).is_file():
            raise ValueError(f'required frozen cache/input missing: {name}')
    sku=json.loads((root/DATA_PATHS[0]).read_text())
    key=json.loads((root/DATA_PATHS[2]).read_text())
    expected=dict(series_ids=[s['series_id'] for s in sku['skus']],
                  test_last_n_days=sku['protocol']['test_last_n_days'])
    if sku.get('locked') is not True or key!=expected:
        raise ValueError('aligned cache key differs from locked SKU protocol; rebuilding forbidden')
    return expected


def required_paths(root,inputs,protocol):
    root=Path(root).resolve()
    cache_contract(root)
    paths=set(DATA_PATHS)|{DESIGN,'requirements.txt',protocol}
    paths.update(p.relative_to(root).as_posix() for folder in ('src','scripts')
                 for p in (root/folder).rglob('*.py'))
    for label in ('calibration','ordering'):
        path=Path(inputs[label]['path'])
        resolved=(root/path).resolve() if not path.is_absolute() else path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('scientific input must be within repository')
        paths.add(resolved.relative_to(root).as_posix())
    for path in paths:
        resolved=(root/path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f'missing or external frozen source: {path}')
    return sorted(paths)


def inventory(root,inputs,protocol):
    root=Path(root).resolve()
    return {path:digest(root/path) for path in required_paths(root,inputs,protocol)}


def validate_coverage(root,study,protocol):
    root=Path(root).resolve()
    required=required_paths(root,study['inputs'],protocol)
    if not set(required)<=set(study['source_hashes']):
        raise ValueError('frozen manifest omits required source/data/protocol files')
    for path in required:
        if digest(root/path)!=study['source_hashes'][path]:
            raise ValueError(f'frozen source changed: {path}')
    for name in ('calibration','ordering'):
        record=study['inputs'][name]
        path=Path(record['path'])
        resolved=(root/path).resolve() if not path.is_absolute() else path.resolve()
        if record['sha256']!=study['source_hashes'][resolved.relative_to(root).as_posix()]:
            raise ValueError('input hash differs from source manifest')
    # Evidence valid only for this call; never cache across scientific units.
    return {path:study['source_hashes'][path] for path in required}


def runtime_identity():
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                packages={name:importlib.metadata.version(name) for name in PACKAGES})
