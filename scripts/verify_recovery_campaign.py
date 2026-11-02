"""Combined final audit of a prospective recovery campaign; read-only.

Requires the externally frozen declaration and plan, not inferred replacements.
Passing archive verification does not imply an economic gate passed or MAPPO.
"""
from pathlib import Path
from scripts.recovery_source_inventory import validate_coverage,runtime_identity
from scripts.verify_recovery_science import check_plan,verify_archive
from scripts.check_recovery_invocations import check as check_invocations


def verify(root,declaration,plan,repository=None):
    if set(declaration)!={'study','protocol','runtime','schedule','overhead'}:
        raise ValueError('complete externally frozen launch declaration required')
    repository=Path.cwd() if repository is None else Path(repository)
    study=declaration['study']
    check_plan(study,plan)
    validate_coverage(repository,study,declaration['protocol'])
    if runtime_identity()!=declaration['runtime']:
        raise ValueError('verification runtime differs from frozen campaign runtime')
    execution=check_invocations(root,declaration,plan)
    if not all(execution.get(key) is True for key in
               ('invocation_accounting_checked','overhead_limits_checked','combined_budget_checked')):
        raise ValueError('complete combined execution-budget verification required')
    science=verify_archive(root,study,plan)
    if science.get('status')!='passed' or science.get('scientific_verification') is not True:
        raise ValueError('complete scientific verification required')
    return dict(status='passed',archive_verified=True,science=science,execution=execution,
                pricing_gate_passed=science['primary_pass'],
                representation_gate_passed=science['representation_pass'],
                heuristic_gate_passed=science['heuristic_pass'],
                ppo_ordering_verified=False,joint_mappo_verified=False,
                full_project_goal_achieved=False,new_simulator_days=0)
