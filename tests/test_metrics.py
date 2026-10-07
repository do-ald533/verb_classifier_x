import torch

from src.metrics import wug_metrics


def test_uniform_logits_give_uniform_regular_prob():
    logits = torch.zeros(10, 20)
    m = wug_metrics(logits, regular_class=0)
    assert abs(m["wug_regular_prob"] - 1 / 20) < 1e-6
    assert m["wug_top2_margin"] < 1e-6


def test_confident_regular_predictions():
    logits = torch.full((10, 20), -10.0)
    logits[:, 0] = 10.0
    m = wug_metrics(logits, regular_class=0)
    assert m["wug_regular_prob"] > 0.99
    assert m["wug_regular_rate"] == 1.0
    assert m["wug_entropy"] < 0.01


def test_wug_regular_prob_is_continuous_not_just_argmax():
    """The whole point of this metric (plan §1.1 #3) is that it varies
    smoothly even when the discrete argmax rate doesn't."""
    logits_a = torch.zeros(10, 20)
    logits_a[:, 0] = 0.1
    logits_b = torch.zeros(10, 20)
    logits_b[:, 0] = 0.5
    prob_a = wug_metrics(logits_a)["wug_regular_prob"]
    prob_b = wug_metrics(logits_b)["wug_regular_prob"]
    rate_a = wug_metrics(logits_a)["wug_regular_rate"]
    rate_b = wug_metrics(logits_b)["wug_regular_rate"]
    assert prob_a != prob_b
    assert rate_a == rate_b == 1.0  # argmax is identical in both cases
