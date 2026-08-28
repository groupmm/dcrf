"""Differentiable Viterbi decoding for linear-chain CRFs.

Accompanies S. Strahl, J. Zeitler, M. Müller: "On the Use of Differentiable Viterbi Decoding for
Linear-Chain CRFs", EUSIPCO 2026.
"""

from .crf import dCRF
from .decode import differentiable_viterbi, nll
from .transitions import DenseTransitions, ToeplitzTransitions

__all__ = ["dCRF", "differentiable_viterbi", "nll", "ToeplitzTransitions", "DenseTransitions"]
