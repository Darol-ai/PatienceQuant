from app.quant_v3.entry_signal import top_k_by_score


def test_top_k_by_score_picks_highest_scoring_symbols():
    scores = {"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.4}

    selected = top_k_by_score(scores, k=2)

    assert selected == {"B", "D"}


def test_top_k_by_score_breaks_ties_by_symbol_ascending():
    scores = {"600003.SH": 0.5, "600001.SH": 0.5, "600002.SH": 0.5, "600004.SH": 0.1}

    selected = top_k_by_score(scores, k=2)

    assert selected == {"600001.SH", "600002.SH"}


def test_top_k_by_score_returns_all_when_fewer_candidates_than_k():
    scores = {"A": 0.2, "B": 0.1}

    selected = top_k_by_score(scores, k=5)

    assert selected == {"A", "B"}


def test_top_k_by_score_handles_empty_scores():
    assert top_k_by_score({}, k=3) == set()
