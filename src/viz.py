"""Shared chart style: consultant-style action titles, one message per chart, recessive
grid, direct labels. Colours are the first three slots of a validated categorical palette
(blue / orange / aqua) plus neutrals; no dual axes; every chart states its source/assumption.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", ink="#0b0b0b", muted="#52514e",
         grid="#e7e6e2", surface="#fcfcfb", neutral="#a7a6a0", blue_light="#cde2fb",
         orange_light="#fbd9cc", red="#c0392b")


def setup():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["muted"], "text.color": C["ink"],
        "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False,
    })


def header(fig, headline, sub=None, top=0.975):
    fig.text(0.02, top, headline, fontsize=15, fontweight="bold", ha="left", va="top", wrap=True)
    if sub:
        fig.text(0.02, top - 0.075, sub, fontsize=10.5, color=C["muted"], ha="left", va="top")


def footnote(fig, text):
    fig.text(0.02, 0.012, text, fontsize=8, color=C["muted"], ha="left", va="bottom")


def save(fig, path, dpi=170):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
