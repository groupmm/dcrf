"""Tests for the decoder, checked against brute-force enumeration of all state sequences."""

import itertools

import pytest
import torch

from dcrf import differentiable_viterbi, nll

K, M = 4, 5


@pytest.fixture
def problem():
    torch.manual_seed(0)
    return torch.randn(2, K, M), torch.randn(K, K)


def path_scores(S, T):
    """Scores sigma of all K ** M state sequences, shape (batch_size, K ** M)."""
    paths = torch.tensor(list(itertools.product(range(K), repeat=M)))
    local = S[..., paths, torch.arange(M)].sum(dim=-1)
    transition = T[paths[:, :-1], paths[:, 1:]].sum(dim=-1)
    return paths, local + transition


def test_hard_decoding_finds_best_path(problem):
    S, T = problem
    paths, scores = path_scores(S, T)
    expected = paths[scores.argmax(dim=-1)]
    assert torch.equal(differentiable_viterbi(S, T, gamma=0.0).argmax(dim=-2), expected)


def test_soft_decoding_approaches_hard(problem):
    S, T = problem
    hard = differentiable_viterbi(S, T, gamma=0.0)
    assert torch.equal(
        differentiable_viterbi(S, T, gamma=1e-3).argmax(dim=-2), hard.argmax(dim=-2)
    )


def test_decoding_is_a_distribution_over_states(problem):
    S, T = problem
    Theta = differentiable_viterbi(S, T, gamma=1.0)
    assert Theta.shape == S.shape
    assert torch.allclose(Theta.sum(dim=-2), torch.ones(2, M))
    assert (differentiable_viterbi(S, T, gamma=0.0).max(dim=-2).values == 1).all()


def test_batch_items_are_independent(problem):
    S, T = problem
    Theta = differentiable_viterbi(S, T, gamma=1.0)
    for b in range(S.shape[0]):
        assert torch.allclose(differentiable_viterbi(S[b : b + 1], T, gamma=1.0), Theta[b : b + 1])


def test_soft_decoding_is_differentiable(problem):
    S, T = problem
    S, T = S.requires_grad_(), T.requires_grad_()
    differentiable_viterbi(S, T, gamma=1.0).sum().backward()
    for grad in (S.grad, T.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_hard_decoding_has_no_gradient(problem):
    S, T = problem
    Theta = differentiable_viterbi(S.requires_grad_(), T.requires_grad_(), gamma=0.0)
    assert not Theta.requires_grad


def test_nll_matches_enumeration(problem):
    S, T = problem
    Q = torch.randint(0, K, (2, M))
    _, scores = path_scores(S, T)
    # itertools.product enumerates the paths in base-K order, so Q indexes into them directly
    flat = (Q * K ** torch.arange(M - 1, -1, -1)).sum(dim=-1, keepdim=True)
    sigma = scores.gather(-1, flat).squeeze(-1)
    expected = (torch.logsumexp(scores, dim=-1) - sigma).mean()
    assert torch.allclose(nll(S, T, Q), expected)
