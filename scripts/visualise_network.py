"""
visualise_network.py — Draw a CTRNN genome as a spatial 2D network map.

Usage
-----
    python scripts/visualise_network.py <run_dir> [--genome PATH] [--out PATH]
                                        [--title STR] [--no-communities]
                                        [--min-weight FLOAT]

Arguments
---------
run_dir         Path to a run directory (contains config.json + best_genome.npz).
--genome PATH   Load a specific .npz genome instead of best_genome.npz.
--out PATH      Save figure to this path (default: network_map.png).
--title STR     Override the figure title.
--no-communities  Skip community detection colouring (faster).
--min-weight FLOAT  Hide edges below this absolute effective weight (default 0.0).

Layout
------
Two side-by-side panels:

  Left  — Neuron type view
          Nodes coloured by type: E (blue), FSI (orange), SII (green).
          Inactive neurons shown as faint crosses.

  Right — Community view
          Active neurons coloured by detected community (greedy modularity).
          Inactive neurons greyed out.

Both panels share the same spatial positions (genome.position in [0,1]²).
Edges are drawn as arrows whose width and transparency encode |weight|.
Excitatory edges are blue, inhibitory red.
Input neurons (☐) and output neurons (★) are marked with extra symbols.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctrnn_evo.logger import load_config, load_genome
from ctrnn_evo.genome import effective_weights


# ── Colour palettes ───────────────────────────────────────────────────────────

TYPE_COLORS  = {0: "#4C8EDA", 1: "#E07A30", 2: "#56A85D"}  # E, FSI, SII
TYPE_LABELS  = {0: "Excitatory (E)", 1: "Fast-Spiking Inh (FSI)", 2: "Slow-Inh (SII)"}
COMM_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#999999",
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
]


# ── Core drawing ──────────────────────────────────────────────────────────────

def _draw_edges(ax, pos, W_eff, active_mask, edge_mask, min_weight, alpha_scale=0.8):
    """Draw directed edges as arrows; width and alpha encode |weight|.

    Only draws edges where edge_mask[i,j] is True and both neurons are active.
    """
    n = len(active_mask)
    active_edges = edge_mask & active_mask[:, None] & active_mask[None, :]
    if not active_edges.any():
        return
    w_max = np.abs(W_eff[active_edges]).max()
    if w_max == 0:
        return

    for i in range(n):
        if not active_mask[i]:
            continue
        for j in range(n):
            if not active_edges[i, j] or i == j:
                continue
            w = float(W_eff[i, j])
            if abs(w) < min_weight:
                continue
            frac  = abs(w) / w_max
            color = "#3A6BC4" if w > 0 else "#C43A3A"   # exc blue / inh red
            lw    = 0.4 + 2.0 * frac
            alpha = 0.15 + alpha_scale * frac
            xi, yi = pos[i]
            xj, yj = pos[j]
            ax.annotate(
                "", xy=(xj, yj), xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    shrinkA=7, shrinkB=7,
                    connectionstyle="arc3,rad=0.08",
                ),
            )


def _draw_nodes(ax, pos, active_mask, node_colors, node_sizes,
                input_idxs, output_idxs, n_max):
    """Draw nodes; mark inputs and outputs with extra glyphs."""
    active = np.where(active_mask)[0]
    inactive = np.where(~active_mask)[0]

    # Inactive: faint grey cross
    for i in inactive:
        x, y = pos[i]
        ax.plot(x, y, "x", color="#CCCCCC", ms=4, mew=0.8, zorder=1)

    # Active: filled circles
    for i in active:
        x, y = pos[i]
        ax.scatter(x, y, s=node_sizes[i], c=[node_colors[i]],
                   edgecolors="white", linewidths=0.6, zorder=3)

    # Input markers (hollow square outline)
    for i in input_idxs:
        if active_mask[i]:
            x, y = pos[i]
            ax.scatter(x, y, s=node_sizes[i] * 2.5, marker="s",
                       facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)

    # Output markers (star outline)
    for i in output_idxs:
        if active_mask[i]:
            x, y = pos[i]
            ax.scatter(x, y, s=node_sizes[i] * 2.5, marker="*",
                       facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)


def _setup_ax(ax, title):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("x position"); ax.set_ylabel("y position")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_facecolor("#F8F8F8")


# ── Main ──────────────────────────────────────────────────────────────────────

def visualise(
    run_dir: Path,
    genome_path: Path | None = None,
    out_path: Path = Path("network_map.png"),
    title: str | None = None,
    communities: bool = True,
    min_weight: float = 0.0,
) -> None:
    cfg, wcfg, _ = load_config(run_dir)

    gpath = genome_path or (run_dir / "best_genome.npz")
    genome = load_genome(gpath)

    pos         = np.array(genome.position)           # [N_max, 2]
    active_mask = np.array(genome.active_mask, bool)  # [N_max]
    edge_mask   = np.array(genome.edge_mask, bool)    # [N_max, N_max]
    ntype       = np.array(genome.neuron_type, int)   # [N_max]
    W_eff       = np.array(effective_weights(genome))  # [N_max, N_max]

    n_max    = cfg.N_max
    n_in     = cfg.n_in
    n_out    = cfg.n_out
    input_idxs  = list(range(n_in))
    output_idxs = list(range(n_max - n_out, n_max))

    n_active = int(active_mask.sum())
    active_idxs = np.where(active_mask)[0]

    # Node sizes — fixed for active, tiny for inactive
    node_sizes = np.where(active_mask, 120, 20).astype(float)

    # ── Build networkx graph for community detection ──────────────────────────
    W_abs = np.abs(W_eff)
    W_sym = (W_abs + W_abs.T) / 2.0
    W_sub = W_sym[np.ix_(active_idxs, active_idxs)]

    G = nx.Graph()
    G.add_nodes_from(range(n_active))
    for i in range(n_active):
        for j in range(i + 1, n_active):
            w = float(W_sub[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    comm_colors_full = ["#AAAAAA"] * n_max   # default grey
    if communities and G.number_of_edges() > 0:
        comms = list(greedy_modularity_communities(G, weight="weight"))
        for ci, comm in enumerate(sorted(comms, key=len, reverse=True)):
            color = COMM_PALETTE[ci % len(COMM_PALETTE)]
            for local_idx in comm:
                global_idx = active_idxs[local_idx]
                comm_colors_full[global_idx] = color
        from networkx.algorithms.community import modularity as nx_modularity
        Q = nx_modularity(G, comms, weight="weight")
        n_comms = len(comms)
    else:
        Q = float("nan")
        n_comms = 0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    run_label = title or f"{run_dir.parent.name}/{run_dir.name[:30]}…"
    genome_label = gpath.name
    fig.suptitle(
        f"{run_label}  |  {genome_label}\n"
        f"Active: {n_active}/{n_max}   "
        f"Edges: {G.number_of_edges()}   "
        f"Q = {Q:.3f}   Communities: {n_comms}",
        fontsize=10, y=1.01,
    )

    # ── Panel 1: neuron type ──────────────────────────────────────────────────
    type_colors_full = [TYPE_COLORS.get(int(ntype[i]), "#888888") for i in range(n_max)]
    _setup_ax(ax1, "Neuron Type")
    _draw_edges(ax1, pos, W_eff, active_mask, edge_mask, min_weight)
    _draw_nodes(ax1, pos, active_mask, type_colors_full, node_sizes,
                input_idxs, output_idxs, n_max)

    legend_elements = [
        mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_LABELS[t])
        for t in sorted(TYPE_COLORS)
    ] + [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=8, label=f"Input ({n_in})"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label=f"Output ({n_out})"),
        Line2D([0],[0], color="#3A6BC4", lw=1.5, label="Excitatory edge"),
        Line2D([0],[0], color="#C43A3A", lw=1.5, label="Inhibitory edge"),
        Line2D([0],[0], marker="x", color="#CCCCCC", lw=0, markersize=6,
               label="Inactive neuron"),
    ]
    ax1.legend(handles=legend_elements, fontsize=7.5, loc="lower right",
               framealpha=0.9, borderpad=0.8)

    # ── Panel 2: communities ──────────────────────────────────────────────────
    _setup_ax(ax2, f"Community Structure  (Q = {Q:.3f}, {n_comms} communities)")
    _draw_edges(ax2, pos, W_eff, active_mask, edge_mask, min_weight, alpha_scale=0.6)
    _draw_nodes(ax2, pos, active_mask, comm_colors_full, node_sizes,
                input_idxs, output_idxs, n_max)

    if communities and n_comms > 0:
        comm_sizes = [len(c) for c in sorted(comms, key=len, reverse=True)]
        comm_patches = [
            mpatches.Patch(
                color=COMM_PALETTE[ci % len(COMM_PALETTE)],
                label=f"Community {ci+1}  (n={comm_sizes[ci]})",
            )
            for ci in range(n_comms)
        ]
        ax2.legend(handles=comm_patches, fontsize=7.5, loc="lower right",
                   framealpha=0.9, borderpad=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise a CTRNN genome as a 2D network map.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing config.json")
    parser.add_argument("--genome", type=Path, default=None,
                        help="Path to a specific .npz genome (default: best_genome.npz)")
    parser.add_argument("--out", type=Path, default=Path("network_map.png"),
                        help="Output image path (default: network_map.png)")
    parser.add_argument("--title", type=str, default=None, help="Override figure title")
    parser.add_argument("--no-communities", action="store_true",
                        help="Skip community detection (faster)")
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="Hide edges below this |weight| (default: 0.0)")
    args = parser.parse_args()

    visualise(
        run_dir=args.run_dir,
        genome_path=args.genome,
        out_path=args.out,
        title=args.title,
        communities=not args.no_communities,
        min_weight=args.min_weight,
    )


if __name__ == "__main__":
    main()
