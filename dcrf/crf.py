"""The dCRF module: a linear-chain CRF with differentiable Viterbi decoding."""

import torch

from .decode import differentiable_viterbi


class dCRF(torch.nn.Module):
    """Maps a score matrix to a decoded state sequence (Eq. 16).

    Wrapper around :func:`dcrf.decode.differentiable_viterbi` and a transition
    parameterization; both can also be used on their own.

    Parameters
    ----------
    transitions : torch.nn.Module
        Module returning the transition matrix T of shape (K, K), e.g.
        :class:`dcrf.transitions.ToeplitzTransitions`.
    gamma : float
        Temperature hyperparameter; for gamma -> 0, decoding approaches hard Viterbi.
    hard_eval : bool
        If True, hard Viterbi decoding (gamma = 0) is used in eval mode, where differentiability
        is typically not necessary. If False (default), the same gamma is used in both modes.
    """

    def __init__(self, transitions, gamma=1.0, hard_eval=False):
        super().__init__()
        self.transitions = transitions
        self.register_buffer("gamma", torch.as_tensor(float(gamma)))
        self.hard_eval = hard_eval

    @property
    def n_states(self):
        return self.transitions.n_states

    @property
    def T(self):
        """Transition matrix T, shape (K, K)."""
        return self.transitions()

    def forward(self, scores):
        """Decodes scores of shape (batch_size, M, K) into Theta of the same shape."""
        gamma = 0.0 if self.hard_eval and not self.training else self.gamma
        Theta = differentiable_viterbi(scores.transpose(-1, -2), self.T, gamma)
        return Theta.transpose(-1, -2).clamp(min=0.0, max=1.0)
