from prometheus.policy.tepe import TEPE


def test_tepe_prioritizes_and_updates():
    tepe = TEPE()
    tepe.add_hypothesis("h1", expected_gain=0.02, expected_runtime=5, overfit_risk=0.1)
    tepe.add_hypothesis("h2", expected_gain=0.01, expected_runtime=2, overfit_risk=0.2)

    nxt = tepe.get_next()
    assert nxt is not None
    tepe.record_result(nxt.name, gain_observed=0.01, runtime_observed=3, overfit_flag=False)

    board = tepe.leaderboard()
    assert len(board) == 2
    assert all("score" in row for row in board)
