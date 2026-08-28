"""Tests for the transition parameterizations and the dCRF module."""

import pytest
import torch

from dcrf import DenseTransitions, ToeplitzTransitions, dCRF

K, M = 6, 8


def test_toeplitz_is_constant_along_diagonals():
    T = ToeplitzTransitions(K)()
    for offset in range(-K + 1, K):
        assert T.diagonal(offset).unique().numel() == 1


def test_toeplitz_uses_the_given_diagonals():
    diagonals = torch.arange(2 * K - 1).float()
    T = ToeplitzTransitions(K, diagonals=diagonals)()
    states = torch.arange(K)
    assert torch.equal(T, diagonals[states[:, None] - states[None, :] + K - 1])


def test_toeplitz_copies_the_given_diagonals():
    diagonals = torch.zeros(2 * K - 1)
    transitions = ToeplitzTransitions(K, diagonals=diagonals)
    diagonals[0] = 99.0
    assert transitions.diagonals[0] == 0.0


def test_toeplitz_rejects_wrong_number_of_diagonals():
    with pytest.raises(ValueError):
        ToeplitzTransitions(K, diagonals=torch.zeros(K))


@pytest.mark.parametrize("cls", [ToeplitzTransitions, DenseTransitions])
@pytest.mark.parametrize("trainable", [False, True])
def test_trainable_toggles_gradients(cls, trainable):
    transitions = cls(K, trainable=trainable)
    assert transitions().shape == (K, K)
    assert all(p.requires_grad == trainable for p in transitions.parameters())


def test_dcrf_preserves_shape_and_range():
    model = dCRF(ToeplitzTransitions(K))
    Theta = model(torch.randn(2, M, K))
    assert Theta.shape == (2, M, K)
    assert (Theta >= 0).all() and (Theta <= 1).all()
    assert model.n_states == K and model.T.shape == (K, K)


def test_hard_eval_switches_on_eval_mode():
    S = torch.randn(2, M, K)
    model = dCRF(ToeplitzTransitions(K), hard_eval=True)
    soft = model(S)
    hard = model.eval()(S)
    assert (hard.max(dim=-1).values == 1).all()
    assert not torch.equal(soft, hard)


def test_same_gamma_in_both_modes_by_default():
    S = torch.randn(2, M, K)
    model = dCRF(ToeplitzTransitions(K))
    assert torch.equal(model.train()(S), model.eval()(S))
