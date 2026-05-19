#!/usr/bin/env python3
"""
Visualise one episode of an evolved CTRNN agent.

Usage
-----
  python scripts/visualise_agent.py <run_dir>  [--seed 0] [--stride 8] [--out agent.gif]
  python scripts/visualise_agent.py            # auto-picks best run in runs/m8/
  python scripts/visualise_agent.py --runs runs/m8/baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctrnn_evo.logger import load_genome, load_config, load_history
from ctrnn_evo.world import WorldConfig, WorldState, sensor_readout, step_world, reset_world
from ctrnn_evo.forward import forward_pass


# ── Rollout ────────────────────────────────────────────────────────────────────

def record_episode(genome, cfg, wcfg, key):
    """Run one episode via JAX lax.scan and return full trajectory as numpy arrays."""
    k_world, k_steps = jax.random.split(key)
    state     = reset_world(k_world, wcfg)
    v0        = jnp.zeros(cfg.N_max, dtype=jnp.float32)
    step_keys = jax.random.split(k_steps, wcfg.episode_steps)

    def body(carry, rng_key):
        world_state, v = carry
        sensors   = sensor_readout(world_state, wcfg)
        input_vec = jnp.zeros(cfg.N_max).at[:cfg.n_in].set(sensors)
        v_new, output, _ = forward_pass(genome, v, input_vec, cfg)
        new_world = step_world(world_state, output, wcfg)
        alive = new_world.agent_energy > 0.0
        rec = {
            "pos":      world_state.agent_pos,                       # [2]
            "energy":   world_state.agent_energy,                    # scalar
            "hotspots": world_state.hotspot_pos,                     # [n_food, 2]
            "activity": jnp.tanh(v_new) * genome.active_mask,       # [N_max]
            "action":   output,                                      # [2]
            "alive":    alive,                                       # bool scalar
        }
        return (new_world, v_new), rec

    (_, _), traj_jax = jax.lax.scan(body, (state, v0), step_keys)
    return {k: np.array(v) for k, v in traj_jax.items()}


# ── Auto-find best run ─────────────────────────────────────────────────────────

def find_best_run(runs_root: Path) -> Path | None:
    best_dir, best_fit = None, -1.0
    for cond_dir in sorted(runs_root.iterdir()):
        if not cond_dir.is_dir():
            continue
        for run_dir in sorted(cond_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "best_genome.npz").exists():
                continue
            hist = load_history(run_dir)
            if not hist:
                continue
            fit = hist[-1]["max_fitness"]
            if fit > best_fit:
                best_fit, best_dir = fit, run_dir
    return best_dir


# ── Animation ─────────────────────────────────────────────────────────────────

def make_animation(traj, genome, cfg, wcfg, stride: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from matplotlib.gridspec import GridSpec

    T     = traj["pos"].shape[0]
    ARENA = wcfg.arena_size
    SIGMA = wcfg.hotspot_sigma
    frames = list(range(0, T, stride))

    active_idxs = np.where(np.array(genome.active_mask, dtype=bool))[0]
    n_active    = len(active_idxs)
    neuron_type = np.array(genome.neuron_type)[active_idxs]  # 0=E, 1=FSI, 2=SII

    # Color per neuron type: E=salmon, FSI=teal, SII=gold
    type_colors = np.array(["#ff6b6b", "#4ecdc4", "#ffe66d"])
    bar_colors  = type_colors[neuron_type.astype(int)]

    TRAIL    = 120
    HIST_LEN = 80   # rows in the rolling neural heatmap

    fig = plt.figure(figsize=(13, 5.5), facecolor="#111111")
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1],
                   hspace=0.45, wspace=0.32,
                   left=0.05, right=0.97, top=0.93, bottom=0.09)
    ax_arena  = fig.add_subplot(gs[:, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_neural = fig.add_subplot(gs[1, 1])

    # ── Arena ──────────────────────────────────────────────────────────────────
    ax_arena.set_facecolor("#0d1117")
    ax_arena.set_xlim(0, ARENA)
    ax_arena.set_ylim(0, ARENA)
    ax_arena.set_aspect("equal")
    ax_arena.set_title("Evolved agent — 2D foraging arena", color="white", fontsize=10)
    ax_arena.tick_params(colors="#555", labelsize=7)
    for sp in ax_arena.spines.values():
        sp.set_edgecolor("#333")

    food_halos  = [plt.Circle((0, 0), SIGMA * 3, color="#00cc55",
                               alpha=0.12, zorder=2) for _ in range(wcfg.n_food)]
    food_dots   = [ax_arena.plot([], [], "o", color="#00ff88",
                                 ms=5, zorder=4, alpha=0.9)[0] for _ in range(wcfg.n_food)]
    for h in food_halos:
        ax_arena.add_patch(h)

    trail_line, = ax_arena.plot([], [], "-",  color="#4488ff", alpha=0.35, lw=1.3, zorder=3)
    agent_dot,  = ax_arena.plot([], [], "o",  color="#88bbff", ms=8,      zorder=5,
                                markeredgecolor="white", markeredgewidth=0.5)
    # Arrow for velocity
    arrow_quiv = ax_arena.quiver([], [], [], [], color="#ffffff", scale=20,
                                 width=0.003, zorder=6, alpha=0.7)

    info_text = ax_arena.text(
        0.02, 0.97, "", transform=ax_arena.transAxes,
        color="white", fontsize=7.5, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8),
    )

    # Legend for neuron types (static)
    from matplotlib.patches import Patch
    legend_elems = [Patch(color="#ff6b6b", label="E (excitatory)"),
                    Patch(color="#4ecdc4", label="FS-I (fast inhibitory)"),
                    Patch(color="#ffe66d", label="SI-I (slow inhibitory)")]
    ax_arena.legend(handles=legend_elems, loc="lower right",
                    fontsize=6.5, framealpha=0.5, facecolor="#222",
                    labelcolor="white", edgecolor="#444")

    # ── Energy panel ──────────────────────────────────────────────────────────
    ax_energy.set_facecolor("#0d1117")
    ax_energy.set_xlim(0, T)
    ax_energy.set_ylim(-0.05, 1.1)
    ax_energy.set_title("Energy over episode", color="white", fontsize=9)
    ax_energy.tick_params(colors="#555", labelsize=7)
    ax_energy.axhline(0.0, color="#444", lw=0.7, ls="--")
    for sp in ax_energy.spines.values():
        sp.set_edgecolor("#333")
    ax_energy.plot(np.arange(T), traj["energy"], color="#2a2a2a", lw=1.0)  # ghost trace
    en_line, = ax_energy.plot([], [], color="#00ff88", lw=1.5)
    en_dot,  = ax_energy.plot([], [], "o", color="#00ff88", ms=4, zorder=5)

    # ── Neural activity panel ─────────────────────────────────────────────────
    ax_neural.set_facecolor("#0d1117")
    act_history = np.zeros((HIST_LEN, n_active))
    neural_im   = ax_neural.imshow(
        act_history, aspect="auto", vmin=-1.0, vmax=1.0,
        cmap="coolwarm", interpolation="nearest", origin="upper",
        extent=[0, n_active, 0, HIST_LEN],
    )
    ax_neural.set_title(f"Neural activity ({n_active} active neurons)", color="white", fontsize=9)
    ax_neural.tick_params(colors="#555", labelsize=7)
    ax_neural.set_xlabel("neuron index", color="#666", fontsize=7)
    ax_neural.set_ylabel(f"← last {HIST_LEN} steps", color="#666", fontsize=7)
    for sp in ax_neural.spines.values():
        sp.set_edgecolor("#333")

    # Colour-coded xtick markers for neuron type
    ax_neural.set_xticks([])

    # ── Update function ────────────────────────────────────────────────────────
    hist_buf = np.zeros((HIST_LEN, n_active))

    def update(fi):
        t = frames[fi]
        alive = bool(traj["alive"][t])

        # --- Arena ---
        t0 = max(0, t - TRAIL)
        trail_line.set_data(traj["pos"][t0:t+1, 0], traj["pos"][t0:t+1, 1])
        px, py = float(traj["pos"][t, 0]), float(traj["pos"][t, 1])
        agent_dot.set_data([px], [py])
        agent_dot.set_color("#88bbff" if alive else "#ff4444")

        # Action arrow
        ax_val = traj["action"][t]
        arrow_quiv.set_offsets(np.array([[px, py]]))
        arrow_quiv.set_UVC(np.array([ax_val[0]]), np.array([ax_val[1]]))

        for i, (halo, dot) in enumerate(zip(food_halos, food_dots)):
            hx, hy = float(traj["hotspots"][t, i, 0]), float(traj["hotspots"][t, i, 1])
            halo.center = (hx, hy)
            dot.set_data([hx], [hy])

        e = float(traj["energy"][t])
        info_text.set_text(
            f"step  {t:4d} / {T}\n"
            f"energy  {e:.3f}\n"
            f"vx {ax_val[0]:+.2f}  vy {ax_val[1]:+.2f}"
        )

        # --- Energy ---
        en_line.set_data(np.arange(t + 1), traj["energy"][:t+1])
        en_dot.set_data([t], [e])

        # --- Neural (rolling buffer: shift up, insert new row at bottom) ---
        hist_buf[:-1] = hist_buf[1:]
        hist_buf[-1]  = traj["activity"][t, active_idxs]
        neural_im.set_data(hist_buf)

        return (trail_line, agent_dot, arrow_quiv,
                *food_halos, *food_dots,
                info_text, en_line, en_dot, neural_im)

    ani = manim.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=True,
    )
    return ani, fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise one evolved CTRNN agent episode.")
    p.add_argument("run_dir", nargs="?", type=Path,
                   help="Path to a run directory (contains best_genome.npz + config.json)")
    p.add_argument("--seed",   type=int, default=0,   help="Episode RNG seed (default 0)")
    p.add_argument("--stride", type=int, default=8,   help="Render every Nth step (default 8)")
    p.add_argument("--out",    type=str, default="agent_episode.gif",
                   help="Output filename (default agent_episode.gif)")
    p.add_argument("--runs",   type=Path,
                   default=Path(__file__).parent.parent / "runs" / "m8",
                   help="Runs root for auto-discovery (default runs/m8/)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            sys.exit(f"Run directory not found: {run_dir}")
    else:
        print("No run_dir given — searching for best run under", args.runs)
        run_dir = find_best_run(args.runs)
        if run_dir is None:
            sys.exit("No complete runs found.")
        print(f"  Selected: {run_dir}")

    print("Loading genome and config...")
    cfg, wcfg, _ = load_config(run_dir)
    genome       = load_genome(run_dir / "best_genome.npz")

    active_n = int(np.array(genome.active_mask).sum())
    print(f"  N_active={active_n}  lambda_conn={cfg.lambda_conn}")

    print(f"Running episode (seed={args.seed})...")
    key  = jax.random.PRNGKey(args.seed)
    traj = record_episode(genome, cfg, wcfg, key)
    T    = traj["pos"].shape[0]
    survived = int(traj["alive"].sum())
    print(f"  Survived {survived}/{wcfg.episode_steps} steps")

    print(f"Building animation (stride={args.stride}, {T // args.stride} frames)...")
    ani, fig = make_animation(traj, genome, cfg, wcfg, stride=args.stride)

    out_path = Path(args.out)
    print(f"Saving to {out_path}  (this may take ~30s)...")
    from matplotlib.animation import PillowWriter
    ani.save(str(out_path), writer=PillowWriter(fps=20), dpi=90)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Done. {out_path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
