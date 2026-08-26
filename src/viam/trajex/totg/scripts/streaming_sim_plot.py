#!/usr/bin/env python3
"""
Plot a streaming session simulation result CSV.

Reads the CSV emitted by viam-trajex-totg-streaming-sim. Renders a single
figure: (W_c x W_r) grid with a viridis colormap over rebase count, a bold
contour at rebases == 0 (the rebase-free frontier), and starved cells
overlaid in red and annotated with the waypoint index where starvation hit.

Numpy + matplotlib only.

Usage:
    streaming_sim_plot.py <input.csv> <output.png>
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def load_csv(path):
    """Return (metadata_dict, list-of-rows).

    Header is `#`-prefixed `key: value` lines preceding the column header.
    Rows are dicts with `commit_window`, `replan_budget`, `rebases`, and
    `starved_at_waypoint` (None if completed).
    """
    metadata = {}
    rows = []
    with open(path, newline='') as f:
        # Pull leading `#` lines for metadata; everything after goes to csv.reader.
        body_lines = []
        for line in f:
            if line.startswith('#'):
                stripped = line[1:].strip()
                if ':' in stripped:
                    key, _, value = stripped.partition(':')
                    metadata[key.strip()] = value.strip()
            else:
                body_lines.append(line)
        reader = csv.DictReader(body_lines)
        for row in reader:
            starved = row['starved_at_waypoint']
            rows.append({
                'commit_window': float(row['commit_window']),
                'replan_budget': float(row['replan_budget']),
                'rebases': int(row['rebases']),
                'starved_at_waypoint': int(starved) if starved else None,
            })
    return metadata, rows


def assemble_grid(rows):
    """Pivot the cell rows into 2D arrays indexed by (W_r, W_c).

    Returns:
        w_c_axis (1D), w_r_axis (1D), rebases (2D shape (n_r, n_c)),
        starved (2D shape (n_r, n_c), np.nan for not-starved).
    """
    w_c_vals = sorted({r['commit_window'] for r in rows})
    w_r_vals = sorted({r['replan_budget'] for r in rows})
    w_c_index = {v: i for i, v in enumerate(w_c_vals)}
    w_r_index = {v: i for i, v in enumerate(w_r_vals)}

    rebases = np.full((len(w_r_vals), len(w_c_vals)), np.nan)
    starved = np.full((len(w_r_vals), len(w_c_vals)), np.nan)
    for r in rows:
        i = w_r_index[r['replan_budget']]
        j = w_c_index[r['commit_window']]
        rebases[i, j] = r['rebases']
        if r['starved_at_waypoint'] is not None:
            starved[i, j] = r['starved_at_waypoint']

    return np.array(w_c_vals), np.array(w_r_vals), rebases, starved


def cell_edges(centers):
    """Compute pcolormesh edge coordinates from cell-center values on a
    geometric grid. Edges sit at the geometric midpoints between centers;
    the outer edges extend by the same log-step the centers use.
    """
    log = np.log(centers)
    mids = (log[:-1] + log[1:]) / 2.0
    first = log[0] - (log[1] - log[0]) / 2.0
    last = log[-1] + (log[-1] - log[-2]) / 2.0
    return np.exp(np.concatenate([[first], mids, [last]]))


def plot(metadata, w_c_axis, w_r_axis, rebases, starved, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    w_c_edges = cell_edges(w_c_axis)
    w_r_edges = cell_edges(w_r_axis)

    # Background: rebase count. Use vmin=0 so zero-rebase cells sit at the
    # darkest end of viridis; starved cells get overlaid separately.
    rebases_for_plot = np.where(np.isnan(starved), rebases, np.nan)
    vmax = max(1, int(np.nanmax(rebases)) if np.any(~np.isnan(rebases)) else 1)
    mesh = ax.pcolormesh(
        w_c_edges, w_r_edges, rebases_for_plot,
        cmap='viridis', vmin=0, vmax=vmax, shading='flat',
    )

    # Starved cells: solid red overlay with the starve waypoint annotated.
    starved_mask = ~np.isnan(starved)
    if np.any(starved_mask):
        red_cmap = ListedColormap([(0.85, 0.15, 0.15, 1.0)])
        ax.pcolormesh(
            w_c_edges, w_r_edges,
            np.where(starved_mask, 1.0, np.nan),
            cmap=red_cmap, shading='flat',
        )
        for i in range(starved.shape[0]):
            for j in range(starved.shape[1]):
                if starved_mask[i, j]:
                    ax.text(
                        w_c_axis[j], w_r_axis[i],
                        f'{int(starved[i, j])}',
                        ha='center', va='center',
                        color='white', fontsize=7, fontweight='bold',
                    )

    # Bold contour at rebases == 0 (the rebase-free frontier). pcolormesh
    # uses edges, but contour wants centers; draw it on the centers grid.
    # Mask out starved cells so the contour isn't tugged by red region values.
    rebase_field = np.where(starved_mask, np.nan, rebases)
    if np.any(rebase_field == 0) and np.any(rebase_field > 0):
        ax.contour(
            w_c_axis, w_r_axis, rebase_field,
            levels=[0.5], colors='white', linewidths=2.5,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('commit window W_c (seconds)')
    ax.set_ylabel('replan budget W_r (seconds)')

    title_bits = [metadata.get('workload', 'unknown workload')]
    if 'n_waypoints' in metadata:
        title_bits.append(f"N={metadata['n_waypoints']}")
    if 'batch_size' in metadata:
        title_bits.append(f"batch={metadata['batch_size']}")
    if 'speed_factor' in metadata:
        title_bits.append(f"speed={metadata['speed_factor']}")
    if 'sample_rate_hz' in metadata:
        title_bits.append(f"{metadata['sample_rate_hz']}Hz")
    ax.set_title('  '.join(title_bits))

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('rebases (red = starved, label = starve waypoint)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'wrote {output_path}', file=sys.stderr)


def main():
    if len(sys.argv) != 3:
        print('usage: streaming_sim_plot.py <input.csv> <output.png>', file=sys.stderr)
        sys.exit(2)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    metadata, rows = load_csv(input_path)
    if not rows:
        print('error: no rows in input CSV', file=sys.stderr)
        sys.exit(1)
    w_c_axis, w_r_axis, rebases, starved = assemble_grid(rows)
    plot(metadata, w_c_axis, w_r_axis, rebases, starved, output_path)


if __name__ == '__main__':
    main()
