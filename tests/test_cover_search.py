import numpy as np
import pytest
from scripts.diagnose_cover_search import CoverPolicy,select_candidate


def test_cover_actions_respect_inventory_position_and_grid():
    obs=np.array([10.,15.,5.,0.,10.],dtype=np.float32)
    assert CoverPolicy(1,2,6).act(obs)[0].tolist()==[6]
    assert CoverPolicy(1,2,4).act(obs)[0].tolist()==[2]
    assert CoverPolicy(1,2,0).act(obs)[0].tolist()==[0]
    assert CoverPolicy(1,2,100).act(obs)[0].tolist()==[40]
    with pytest.raises(ValueError):CoverPolicy(1,2,float('nan'))


def test_training_selection_service_guard_and_deterministic_ties():
    rows=[dict(cover=4,reward_day=100,fill=.59),dict(cover=5.5,reward_day=80,fill=.6),
          dict(cover=6.5,reward_day=80,fill=.6),dict(cover=6,reward_day=70,fill=.6)]
    assert select_candidate(rows,.6)['cover']==5.5
    rows.append(dict(cover=6,reward_day=80,fill=.6))
    assert select_candidate(rows,.6)['cover']==6
    with pytest.raises(ValueError):select_candidate([], .6)
