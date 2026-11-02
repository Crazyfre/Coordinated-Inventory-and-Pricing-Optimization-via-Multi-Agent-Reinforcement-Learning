"""Training-only action-independent reference baseline for joint MAPPO.

The simulator and stored economic rewards are never changed. Paired mode
predicts residual values and applies existing interval GAE to learner minus
frozen reference rewards. See the prospective protocol for the exact MC identity
and its limitations under approximate GAE, clipping and advantage normalization.
"""
import numpy as np
from src.algos.action_marginal_mappo import ActionMarginalMAPPO


class ReferenceBaselineMAPPO(ActionMarginalMAPPO):
    def __init__(self,*args,reference_baseline=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference_baseline=bool(reference_baseline)

    def update(self,episodes,*,epochs=4):
        if not self.reference_baseline:return super().update(episodes,epochs=epochs)
        residual=[]
        for ep in episodes:
            original=np.asarray(ep['rewards'],dtype=np.float64)
            reference=np.asarray(ep['reference_rewards'],dtype=np.float64)
            if original.shape!=(28,) or reference.shape!=(28,) or not np.isfinite(reference).all():
                raise ValueError('28 finite frozen reference rewards required')
            # Shallow episode copy; rows/trace stay read-only in inherited update.
            residual.append(dict(ep,rewards=(original-reference).tolist()))
        return super().update(residual,epochs=epochs)

    def checkpoint(self):
        data=super().checkpoint()
        data.update(variant='reference_baseline_structured_mappo_v1',reference_baseline=self.reference_baseline)
        return data

    def restore(self,data):
        if data['variant']!='reference_baseline_structured_mappo_v1' or data['reference_baseline']!=self.reference_baseline:
            raise ValueError('matching reference-baseline checkpoint required')
        super().restore(dict(data,variant='action_marginal_structured_mappo_v1'))
