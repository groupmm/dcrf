"""Differentiable Viterbi decoding and the negative log-likelihood of a linear-chain CRF.

Notation follows S. Strahl, J. Zeitler, M. Müller: "On the Use of Differentiable Viterbi
Decoding for Linear-Chain CRFs", EUSIPCO 2026. Score matrix S (K states, M frames),
transition matrix T, temperature gamma, subsequence scores A, backtracking probabilities B,
reference state sequence Q, and decoded sequence Theta. The functions below take these as
``scores``, ``transitions`` and ``states``, and alias them to the paper's symbols so that the
implementation can be read next to the equations. All tensors carry a leading batch dimension.
"""

import torch
import torch.nn.functional as F


def differentiable_viterbi(scores, transitions, gamma=1.0):
    """Differentiable Viterbi decoding (Algorithm 1).

    The max operations of the Viterbi algorithm are replaced by their temperature-controlled
    counterparts max^gamma (Eq. 3) and argmax^gamma (Eq. 10), so that the decoded sequence stays
    differentiable w.r.t. the scores and the transition matrix.

    Parameters
    ----------
    scores : torch.Tensor
        Score matrix S, shape (batch_size, K, M).
    transitions : torch.Tensor
        Transition matrix T, shape (K, K).
    gamma : float or torch.Tensor
        Temperature hyperparameter. For gamma -> 0 decoding approaches hard Viterbi;
        gamma = 0 selects it exactly, in which case the result is one-hot and has no gradient.

    Returns
    -------
    torch.Tensor
        Decoded sequence Theta, shape (batch_size, K, M).
    """
    S, T = scores, transitions
    K, M = S.shape[-2:]
    soft = float(gamma) != 0.0

    if soft:
        # max^gamma(d) = gamma * logsumexp(d / gamma), so scaling S and T once lets the
        # recursion run on A / gamma, from which argmax^gamma follows as a softmax
        S = S / gamma
        T = T / gamma

    # forward recursion, keeping only B (A is needed at the previous step only)
    B = []
    A = S[..., 0]  # Eq. 6
    for m in range(1, M):
        step = T + A.unsqueeze(-1)  # A(:, m - 1) + T(:, k), shape (batch, k, k')
        if soft:
            B.append(F.softmax(step, dim=-2))  # Eq. 9
            A = torch.logsumexp(step, dim=-2) + S[..., m]  # Eq. 7
        else:
            A, idxs = step.max(dim=-2)
            B.append(idxs)
            A = A + S[..., m]

    # backtracking recursion
    if soft:
        Theta = F.softmax(A, dim=-1)  # Eq. 13
        Theta_seq = [Theta]
        for B_m in reversed(B):
            Theta = torch.matmul(B_m, Theta.unsqueeze(-1)).squeeze(-1)  # Eq. 15
            Theta_seq.append(Theta)
        return torch.stack(Theta_seq[::-1], dim=-1)

    # hard decoding keeps Theta one-hot, so backtrack state indices instead
    idxs = A.argmax(dim=-1)
    idxs_seq = [idxs]
    for B_m in reversed(B):
        idxs = B_m.gather(-1, idxs.unsqueeze(-1)).squeeze(-1)
        idxs_seq.append(idxs)
    idxs_seq = torch.stack(idxs_seq[::-1], dim=-1)
    return F.one_hot(idxs_seq, num_classes=K).to(S.dtype).transpose(-1, -2)


def nll(scores, transitions, states):
    """Negative log-likelihood of state sequences under the CRF.

    Parameters
    ----------
    scores : torch.Tensor
        Score matrix S, shape (batch_size, K, M).
    transitions : torch.Tensor
        Transition matrix T, shape (K, K).
    states : torch.Tensor
        Reference state sequences Q, shape (batch_size, M).

    Returns
    -------
    torch.Tensor
        Scalar NLL, averaged over the batch.
    """
    S, T, Q = scores, transitions, states

    # numerator: score sigma(Q, S, T) of the given state sequence (Eq. 1)
    local_scores = S.gather(-2, Q.unsqueeze(-2)).sum(dim=(-2, -1))
    transition_scores = T[Q[:, :-1], Q[:, 1:]].sum(dim=-1)
    sigma = local_scores + transition_scores

    # denominator: log partition function via the forward algorithm
    log_alpha = S[..., 0]
    for m in range(1, S.shape[-1]):
        log_alpha = torch.logsumexp(log_alpha.unsqueeze(-1) + T, dim=-2) + S[..., m]
    log_partition = torch.logsumexp(log_alpha, dim=-1)

    return (log_partition - sigma).mean()
