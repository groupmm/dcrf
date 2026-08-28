"""Parameterizations of the CRF transition matrix T.

Each module takes no input and returns T of shape (K, K), where T[i, j] scores a transition from
state i at frame m to state j at frame m + 1. Any ``torch.nn.Module`` with this signature can be
passed to :class:`dcrf.crf.dCRF`.
"""

import torch


class DenseTransitions(torch.nn.Module):
    """Fully parameterized T (dCRF), initialized standard normal.

    Parameters
    ----------
    n_states : int
        Number of states K.
    trainable : bool
        Whether the transition scores receive gradients.
    """

    def __init__(self, n_states, trainable=False):
        super().__init__()
        self.n_states = n_states
        self.scores = torch.nn.Parameter(torch.randn(n_states, n_states), requires_grad=trainable)

    def forward(self):
        return self.scores


class ToeplitzTransitions(torch.nn.Module):
    """Toeplitz-structured T with a single score per diagonal (dCRF-T).

    Parameters
    ----------
    n_states : int
        Number of states K.
    diagonals : array_like, optional
        The 2 * K - 1 diagonal scores, ordered from the upper left to the lower right corner, so
        that index K - 1 is the main diagonal. Random initialization if None.
    trainable : bool
        Whether the diagonal scores receive gradients.
    """

    def __init__(self, n_states, diagonals=None, trainable=False):
        super().__init__()
        self.n_states = n_states

        if diagonals is None:
            diagonals = torch.randn(2 * n_states - 1)
        else:
            diagonals = torch.as_tensor(diagonals, dtype=torch.get_default_dtype()).clone()
            if diagonals.shape != (2 * n_states - 1,):
                raise ValueError(f"diagonals must have shape ({2 * n_states - 1},)")
        self.diagonals = torch.nn.Parameter(diagonals, requires_grad=trainable)

        # index of the diagonal each (state at m, state at m + 1) pair lies on
        states = torch.arange(n_states)
        self.register_buffer(
            "idxs", states[:, None] - states[None, :] + n_states - 1, persistent=False
        )

    def forward(self):
        return self.diagonals[self.idxs]
