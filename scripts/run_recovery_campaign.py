"""Run one frozen invocation, or independently audit the complete archive.

Supply the externally recorded SHA256 of a packet containing root, declaration
and plan. This entry point never generates scientific parameters or retries.
"""
import time
ENTRY_STARTED = time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path


def read_packet(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('external launch packet checksum mismatch')
    packet = json.loads(raw)
    if set(packet) != {'root', 'declaration', 'plan'}:
        raise ValueError('complete launch packet required')
    root = Path(packet['root'])
    if not root.is_absolute() or not root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('absolute archive root within repository required')
    from scripts.verify_recovery_science import check_plan
    from scripts.recovery_campaign_plan import build
    from scripts.recovery_invocation import validate_schedule
    declaration = packet['declaration']
    if set(declaration) != {'study', 'protocol', 'runtime', 'schedule', 'overhead'}:
        raise ValueError('complete launch declaration required')
    check_plan(declaration['study'], packet['plan'])
    if build(declaration['study']) != packet['plan']:
        raise ValueError('launch plan differs from frozen packet')
    validate_schedule(packet['plan'], declaration['schedule'])
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'verify'))
    parser.add_argument('--packet', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    packet = read_packet(args.packet, args.sha256)
    if args.operation == 'run':
        from scripts.recovery_invocation import launch
        result = launch(packet['root'], packet['declaration'], started_at=ENTRY_STARTED)
    else:
        from scripts.verify_recovery_campaign import verify
        result = verify(packet['root'], packet['declaration'], packet['plan'])
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
