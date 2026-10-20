def test_critic_free_advantages_do_not_query_learned_value():
    from src.algos.ordering_critic_probe import CriticFreeForecastPriorMAPPO
    model = object.__new__(CriticFreeForecastPriorMAPPO)
    assert model.value([1,2,3]) == 0.
