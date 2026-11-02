"""Zero-update exact-root-action integration, using unchanged local linear actors.

For one independently chosen SKU per role decision, evaluate all reachable physical
root actions under the frozen joint continuation. The actual root suffix and its
strong replacement are reused from the proof-bound collector. No optimizer fit.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from outputs.linear_coordinate_diagnostic import LinearCoordinateDiagnostic,episode_gradients
from scripts.train_compact_residual import serial


def trace_hash(value):
    return hashlib.sha256(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class EnumeratedCreditDiagnostic(LinearCoordinateDiagnostic):
    def collect(self,env,key,*,budget,deadline):
        hashes={};original=self.continuation;had_override='continuation' in self.__dict__
        def observed(snapshot,role,action,**kwargs):
            first=snapshot.day;rr,trace=original(snapshot,role,action,**kwargs)
            hashes[role,first,tuple(int(x) for x in action)]=trace_hash(trace);return rr,trace
        self.continuation=observed
        try:
            ep=super().collect(env,key,budget=budget,deadline=deadline,keep_proof_snapshots=True)
            terminal=ep['actual_policy_rng'].clone();roots=[]
            try:
                for role,row,snapshot,rng in ep.pop('proof_snapshots'):
                    i=row['coordinate'];actor=getattr(self,role)
                    with torch.no_grad():p=actor.distribution(torch.tensor(row['features'])).probs[i].double().numpy()
                    support=np.flatnonzero(p>0);entries=[];start=row['start'];first=start+1
                    for value in support:
                        action=row['action'].copy();action[i]=value
                        if np.array_equal(action,row['action']):
                            rr=ep['rewards'][start:].copy();digest=trace_hash(ep['trace'][start:]);source='actual'
                        elif np.array_equal(action,row['reference_action']):
                            rr=row['reference_rewards'].copy();digest=hashes[role,first,tuple(int(x) for x in action)];source='strong'
                        else:
                            torch.set_rng_state(rng)
                            rr,trace=self.continuation(copy.deepcopy(snapshot),role,action,budget=budget,deadline=deadline)
                            digest=trace_hash(trace);source='branch'
                        entries.append(dict(action=int(value),probability=float(p[value]),rewards=rr,trace_sha256=digest,source=source,
                            discounted_return=float(np.dot(self.gamma**np.arange(len(rr)),rr)/100)))
                    anchor=int(row['reference_action'][i]);base=next(r['discounted_return'] for r in entries if r['action']==anchor)
                    for entry in entries:entry['residual_return']=entry['discounted_return']-base
                    roots.append(dict(role=role,root_rng=rng.clone(),entries=entries))
            finally:torch.set_rng_state(terminal)
            ep['roots']=roots;return ep
        finally:
            if had_override:self.continuation=original
            else:del self.continuation

    def checkpoint(self):
        return dict(variant='enumerated_credit_diagnostic_v1',gamma=self.gamma,model=self.state_dict(),updates=0,trainable=False)


def gradients(model,ep,whole_reference):
    result=episode_gradients(model,ep,whole_reference)
    for role in ('pricing','ordering'):
        rows=ep['rows'][role];roots=[r for r in ep['roots'] if r['role']==role]
        objective=next(getattr(model,role).parameters()).sum()*0
        for row,root in zip(rows,roots):
            i=row['coordinate'];d=getattr(model,role).distribution(torch.tensor(row['features']))
            p=d.probs[i];values=torch.tensor([r['residual_return'] for r in root['entries']],dtype=torch.float32)
            actions=torch.tensor([r['action'] for r in root['entries']],dtype=torch.long)
            # Exact conditional action expectation at the OLD weights. Detached
            # Q estimates; selected score baseline has zero expected derivative.
            objective=objective+10*model.gamma**row['start']*(p[actions]*values).sum()/len(rows)
        result[role]['enumerated']=torch.autograd.grad(objective,list(getattr(model,role).parameters()))[0].detach().double().flatten().numpy()
    return result


def analytic_enumerated(ep,pricing,ordering,gamma):
    result={}
    for role,weight in (('pricing',pricing),('ordering',ordering)):
        weight=np.asarray(weight,dtype=float);g=np.zeros_like(weight);rows=ep['rows'][role]
        roots=iter(r for r in ep['roots'] if r['role']==role)
        for row in rows:
            root=next(roots);i=row['coordinate'];f=np.asarray(row['features'],dtype=float)[i,:weight.shape[1]]
            z=f@weight[i];z-=z.max();p=np.exp(z);p/=p.sum()
            physical=np.arange(5) if role=='pricing' else np.clip(int(row['features'][i,-1])+np.array([-1,0,1]),0,40)
            for entry in root['entries']:
                mask=physical==entry['action'];mass=float(p[mask].sum());score=p*mask/mass-p
                g[i]+=10*gamma**row['start']*mass*entry['residual_return']*f[:,None]*score/len(rows)
        result[role]=g.ravel()
    return result


def summary(vectors):
    table={}
    for role in ('pricing','ordering'):
        table[role]={};arrays={}
        for kind in ('ordinary','coordinate','enumerated'):
            values=np.asarray([v[role][kind] for v in vectors]);arrays[kind]=values
            mean=values.mean(0);variance=float(values.var(0,ddof=1).sum());norm=float(np.square(mean).sum())
            table[role][kind]=dict(mean_squared_norm=norm,centered_trace_variance=variance,
                estimated_mean_snr=norm*len(values)/variance if variance>0 else None)
        for kind in ('ordinary','coordinate'):
            a=arrays[kind].mean(0);b=arrays['enumerated'].mean(0);den=float(np.linalg.norm(a)*np.linalg.norm(b))
            table[role]['enumerated_vs_'+kind]=dict(mean_cosine=float(np.inner(a,b)/den) if den else None,
                variance_ratio=table[role]['enumerated']['centered_trace_variance']/table[role][kind]['centered_trace_variance']
                if table[role][kind]['centered_trace_variance'] else None)
    return table
