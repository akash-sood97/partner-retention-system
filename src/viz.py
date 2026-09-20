"""Shared chart style: consultant-style action titles, one message per chart, recessive
grid, direct labels. Colours are the first three slots of a validated categorical palette
(blue / orange / aqua) plus neutrals; no dual axes; every chart states its source/assumption.
"""
import textwrap

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


def decision_card(path, title, kpis, draw, note=None, dpi=200):
    """The 'decision in one picture' slide: the decision, three figures, one small chart.

    Always 16:9 at 1600x900, so every project's card is the same shape on the portfolio
    page. It is read at about 300px wide on a phone, so this is composed as a slide and
    not as a chart: nothing below 13pt, and `draw(ax)` should paint one simple shape with
    a handful of direct labels and no legend.

    title  one sentence, the decision itself
    kpis   up to three (value, label) pairs; keep labels under about six words
    draw   callback that paints the chart into the axes it is handed
    note   optional one-line provenance/assumption strip along the bottom
    """
    wrapped = textwrap.fill(title, 58)
    if wrapped.count("\n") > 1:
        raise ValueError(
            f"decision_card title wraps to {wrapped.count(chr(10)) + 1} lines and would "
            f"overprint the figures below it; keep it under ~116 characters:\n{title}")

    fig = plt.figure(figsize=(8, 4.5), dpi=dpi)
    fig.text(0.035, 0.955, wrapped, fontsize=19, fontweight="bold",
             ha="left", va="top", linespacing=1.32)

    for i, (value, label) in enumerate(kpis[:3]):
        y = 0.70 - i * 0.205
        # long values (a before -> after pair) shrink so they never reach the chart
        fig.text(0.035, y, value, fontsize=24 if len(value) <= 14 else 19,
                 fontweight="bold", ha="left", va="top", color=C["ink"])
        fig.text(0.035, y - 0.082, textwrap.fill(label, 32), fontsize=11.5,
                 color=C["muted"], ha="left", va="top", linespacing=1.3)

    # the chart sits clear of the note strip, so tick labels never overprint it
    ax = fig.add_axes([0.43, 0.20 if note else 0.12, 0.54, 0.52 if note else 0.60])
    draw(ax)

    if note:
        fig.text(0.035, 0.022, note, fontsize=9.5, color=C["muted"], ha="left", va="bottom")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
