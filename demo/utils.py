"""Helpers for the dCRF demo notebook: log-frequency axis, VQT front-end, plotting."""

import matplotlib.pyplot as plt
import nnAudio.features
import torch


def cents_to_hz(f_cent, f_ref):
    """Converts frequencies in cents to Hz."""
    return f_ref * 2 ** (f_cent / 1200)


def get_log_frequencies(f_min=55.0, f_max=1760.0, cent_step=10.0):
    """Creates a discrete logarithmic frequency axis."""
    n_freq = int(torch.log2(torch.as_tensor(f_max / f_min)) * 1200 / cent_step)
    f_cent = torch.arange(n_freq) * cent_step
    return cents_to_hz(f_cent, f_ref=f_min)


class VQTModule(torch.nn.Module):
    """VQT with logarithmic compression, yielding emission logits on a log-frequency grid."""

    def __init__(self, log_comp_gamma=0, bins_per_semitone=3, **kwargs):
        super().__init__()

        self.frequencies_hz = get_log_frequencies(
            f_min=kwargs["f0_min"],
            f_max=kwargs["f0_max"],
            cent_step=100 / bins_per_semitone,
        )

        self.vqt = nnAudio.features.VQT(
            sr=kwargs["fs"],
            hop_length=kwargs["hop_size"],
            fmin=kwargs["f0_min"],
            fmax=kwargs["f0_max"],
            n_bins=self.frequencies_hz.numel(),
            filter_scale=kwargs["filter_scale"],
            bins_per_octave=bins_per_semitone * 12,
            norm=kwargs["norm"],
            basis_norm=kwargs["basis_norm"],
            gamma=kwargs["gamma"],
            window=kwargs["window"],
            pad_mode=kwargs["pad_mode"],
            earlydownsample=kwargs["earlydownsample"],
            trainable=kwargs["trainable"],
            output_format=kwargs["output_format"],
            verbose=kwargs["verbose"],
        )

        self.log_comp_gamma = log_comp_gamma

    def forward(self, x):
        x_vqt = self.vqt(x)
        return torch.log(1 + self.log_comp_gamma * x_vqt.permute(0, 2, 1))


def plottable(x):
    return x.detach().cpu().numpy()


def plot_matrix(
    M,
    x,
    y,
    ax,
    cmap="gray_r",
    norm=None,
    xlabel="Time (seconds)",
    ylabel="Frequency (Hz)",
    xscale="linear",
    yscale="log",
    title=None,
    colorbar=False,
):
    """pcolormesh of a (len(y), len(x)) matrix on (possibly logarithmic) coordinate axes."""
    im = ax.pcolormesh(
        plottable(torch.as_tensor(x)),
        plottable(torch.as_tensor(y)),
        M,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=9)
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im
