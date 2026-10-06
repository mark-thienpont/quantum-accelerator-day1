"""Display and plotting helpers for M4 - Nurse rostering.

Kept out of the notebook so the notebook shows the *model*, not the plumbing.
2026 Quantum Accelerator, Day 1.
"""
import numpy as np
import matplotlib.pyplot as plt


def render(grid, req, seniors, title=None):
    """Print a roster grid as a readable table."""
    n_nurses, n_days = grid.shape
    lines = []
    if title:
        lines.append(title)
    lines.append("            " + "".join(f"{d % 10}" for d in range(n_days)))
    for n in range(n_nurses):
        tag = "S" if n in seniors else "j"
        row = "".join("#" if grid[n, d] else "." for d in range(n_days))
        lines.append(f"  nurse {n:02d}{tag} {row}   {int(grid[n].sum())} days")
    lines.append("  coverage   " + "".join(f"{int(grid[:, d].sum()) % 10}" for d in range(n_days)))
    lines.append("  required   " + "".join(f"{int(req[d]) % 10}" for d in range(n_days)))
    print("\n".join(lines))


def show_preferences(pref):
    """Heatmap of 'would rather not work this day'."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(pref, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("day")
    ax.set_ylabel("nurse")
    ax.set_title("Preference matrix: red = would rather not work this day")
    ax.set_xticks(range(pref.shape[1]))
    ax.set_yticks(range(pref.shape[0]))
    plt.tight_layout()
    plt.show()


def plot_cliff(labels, feas_pct, mean_broken):
    """The headline plot: feasibility collapse as constraint families accumulate."""
    stages = np.arange(1, len(labels) + 1)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
    ax1.plot(stages, feas_pct, "o-", lw=2.5, ms=9, color="#1f77b4",
             label="% of annealing runs that are feasible")
    ax1.set_ylabel("feasible runs (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(stages)
    ax1.set_xticklabels([f"{i}\n{l}" for i, l in zip(stages, labels)], fontsize=8)
    ax1.set_xlabel("constraint families active (cumulative)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(stages, mean_broken, "s--", lw=2, ms=7, color="#d62728",
             label="mean broken constraints per run")
    ax2.set_ylabel("mean broken constraints", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9)
    ax1.set_title("The feasibility cliff: same solver, same variables, more constraints")
    plt.tight_layout()
    plt.show()


def plot_lambda_balance(lams, feas_pct):
    """Penalty balance: interior optimum in the weight of one constraint family."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogx(lams, feas_pct, "o-", lw=2.5, ms=9, color="#2ca02c")
    best = int(np.argmax(feas_pct))
    ax.axvline(lams[best], ls=":", color="grey")
    ax.annotate(f"best balance\nlam_rest = {lams[best]:g}",
                xy=(lams[best], feas_pct[best]),
                xytext=(10, -28), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_xlabel("penalty weight on the rest rule (all other families fixed)")
    ax.set_ylabel("feasible runs (%)")
    ax.set_title("Penalty balance, not penalty magnitude")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_effort(sweeps_list, series, labels):
    """How much annealing effort buys back feasibility, per stage."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for row, lab in zip(series, labels):
        ax.semilogx(sweeps_list, row, "o-", lw=2, ms=7, label=lab)
    ax.set_xlabel("num_sweeps (annealing effort per run)")
    ax.set_ylabel("feasible runs (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("More effort buys some of it back, but not all")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_transition(rates, cpsat_times, cap, sa_feasible=None):
    """CP-SAT solve time (log) against conflict rate, with the cap drawn in.

    Points sitting on the dashed cap line were never resolved.
    """
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rates = np.asarray(rates, dtype=float)
    times = np.asarray(cpsat_times, dtype=float)
    for j in range(times.shape[1]):
        ax.semilogy(rates, times[:, j], "o-", lw=1.6, ms=7, alpha=0.85,
                    label=f"CP-SAT, instance {j + 1}")
    ax.axhline(cap, ls="--", color="crimson", lw=1.5)
    ax.text(rates[0], cap * 1.15, f"time limit ({cap:g} s) — points here were never resolved",
            color="crimson", fontsize=8.5, va="bottom")
    ax.set_xlabel("conflict rate: fraction of nurse pairs forbidden to share a day")
    ax.set_ylabel("CP-SAT wall time (s, log scale)")
    ax.set_title("A razor-sharp transition, and huge variance inside it")
    ax.grid(alpha=0.3, which="both")

    if sa_feasible is not None:
        ax2 = ax.twinx()
        ax2.plot(rates, sa_feasible, "s:", color="#444444", lw=2, ms=7,
                 label="annealing: feasible runs (%)")
        ax2.set_ylabel("annealing feasible runs (%)", color="#444444")
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis="y", labelcolor="#444444")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    else:
        ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.show()


def show_conflict_graph(n_nurses, pairs):
    """Draw the 'cannot work the same day' graph."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n_nurses))
    g.add_edges_from(pairs)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    nx.draw(g, pos=nx.circular_layout(g), ax=ax, node_size=230, node_color="#1f77b4",
            edge_color="#999999", with_labels=True, font_size=7, font_color="white")
    ax.set_title(f"{n_nurses} nurses, {len(pairs)} forbidden pairs\n"
                 f"a day's roster must be an independent set in this graph", fontsize=10)
    plt.tight_layout()
    plt.show()
