#!/usr/bin/env python3
"""
Visualize trajectory generation from JSON output.

Usage:
    python scripts/visualize_trajectory.py trajectory.json
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def _f(v):
    """Convert a JSON value (possibly null/None) to float, mapping None to NaN."""
    return float('nan') if v is None else float(v)


# Visualization parameters
LIMIT_CURVE_MARGIN = 1.15
MAX_Y_SCALE_FACTOR = 2.5
SWITCHING_POINT_SIZE = 120
SWITCHING_POINT_ALPHA = 0.7
SWITCHING_POINT_LINE_WIDTH = 1
LIMIT_MARKER_SIZE = 10


def load_trajectory(f):
    """Load trajectory JSON from a file-like object with validation."""
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required top-level keys
    required = ['metadata', 'integration_points', 'events']
    missing = [k for k in required if k not in data]
    if missing:
        print(f"Error: Missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    return data


def plot_phase_plane(data, ax):
    """Plot phase plane trajectory (s, s_dot) with limit curves."""
    # Main trajectory (green like the paper)
    s = np.array([float(x) for x in data['integration_points']['s']])
    s_dot = np.array([float(x) for x in data['integration_points']['s_dot']])

    ax.plot(s, s_dot, 'g-', linewidth=2, label='Trajectory', zorder=3)

    # Build limit curve data by merging integration point samples with any within-range
    # limit_curve_samples (e.g. at pruned point s-values). Points beyond the last
    # integration point s form the gap overlay, rendered dashed.
    last_s = float(s[-1]) if len(s) > 0 else float('inf')

    lcs = data.get('limit_curve_samples')
    lcs_gap_s, lcs_gap_acc, lcs_gap_vel = [], [], []

    merged = list(zip(
        [float(x) for x in data['integration_points']['s']],
        data['integration_points']['s_dot_max_acc'],
        data['integration_points']['s_dot_max_vel'],
    ))
    if lcs:
        for s_val, acc_val, vel_val in zip(lcs['s'], lcs['s_dot_max_acc'], lcs['s_dot_max_vel']):
            s_f = float(s_val)
            if s_f <= last_s:
                merged.append((s_f, acc_val, vel_val))
            else:
                lcs_gap_s.append(s_f)
                lcs_gap_acc.append(acc_val)
                lcs_gap_vel.append(vel_val)

    merged.sort(key=lambda x: x[0])
    s_for_limits = [x[0] for x in merged]
    s_dot_max_acc = [x[1] for x in merged]
    s_dot_max_vel = [x[2] for x in merged]

    # Calculate y-axis limits ensuring trajectory visibility:
    # - Trajectory should occupy at least 60% of vertical space
    # - If limits are very high (e.g., nearly infinite), cap y_max to avoid
    #   squashing the trajectory into an unreadable band at the bottom
    # - Use MAX_Y_SCALE_FACTOR as a heuristic balance
    # Filter inf/nan from s_dot - failed trajectories can have non-finite integration points
    finite_s_dot = s_dot[np.isfinite(s_dot)]
    max_traj_velocity = max(finite_s_dot) if len(finite_s_dot) > 0 else 0.0

    # Collect all limit curve values (merged main data plus gap samples)
    limit_values = []
    for val in s_dot_max_acc:
        if val is not None:
            limit_values.append(float(val))
    for val in s_dot_max_vel:
        if val is not None:
            limit_values.append(float(val))
    for val in lcs_gap_acc:
        if val is not None:
            limit_values.append(float(val))
    for val in lcs_gap_vel:
        if val is not None:
            limit_values.append(float(val))

    if limit_values:
        max_limit = max(limit_values)
        min_limit = min(limit_values)
        y_max = min(max_limit * LIMIT_CURVE_MARGIN, max_traj_velocity * MAX_Y_SCALE_FACTOR)
        # Always show the most restrictive finite limit curve even if the trajectory
        # hasn't reached it yet (common in partial/failed integrations)
        y_max = max(y_max, min_limit * LIMIT_CURVE_MARGIN)
    else:
        y_max = max_traj_velocity * 1.3

    # Add buffer below trajectory
    y_min = -0.05 * max_traj_velocity

    # Acceleration limit curve (red - primary boundary)
    # Track transitions to show vertical bars
    prev_was_inf = True
    prev_s = None
    prev_val = None
    for i, val in enumerate(s_dot_max_acc):
        curr_s = float(s_for_limits[i])
        if val is not None:
            curr_val = float(val)
            # Show vertical drop when transitioning from infinite to finite
            if prev_was_inf and i > 0:
                ax.plot([curr_s, curr_s], [y_max, curr_val], 'r-',
                       linewidth=1, alpha=0.5, zorder=1)
            prev_was_inf = False
            prev_s = curr_s
            prev_val = curr_val
        else:
            # Show vertical rise when transitioning from finite to infinite
            if not prev_was_inf and prev_s is not None and prev_val is not None:
                ax.plot([prev_s, prev_s], [prev_val, y_max], 'r-',
                       linewidth=1, alpha=0.5, zorder=1)
            prev_was_inf = True

    # Plot the actual limit curve segments
    segments_acc = []
    current_segment_s = []
    current_segment_v = []
    for i, val in enumerate(s_dot_max_acc):
        if val is not None:
            current_segment_s.append(float(s_for_limits[i]))
            current_segment_v.append(float(val))
        else:
            if current_segment_s:
                segments_acc.append((current_segment_s, current_segment_v))
                current_segment_s = []
                current_segment_v = []
    if current_segment_s:
        segments_acc.append((current_segment_s, current_segment_v))

    for i, (seg_s, seg_v) in enumerate(segments_acc):
        label = 'Acceleration Limit' if i == 0 else None
        ax.plot(seg_s, seg_v, 'r-', linewidth=1.5, label=label, zorder=2)

    # Velocity limit curve (orange - secondary boundary)
    prev_was_inf = True
    prev_s = None
    prev_val = None
    for i, val in enumerate(s_dot_max_vel):
        curr_s = float(s_for_limits[i])
        if val is not None:
            curr_val = float(val)
            # Show vertical drop when transitioning from infinite to finite
            if prev_was_inf and i > 0:
                ax.plot([curr_s, curr_s], [y_max, curr_val], 'orange',
                       linewidth=1, alpha=0.5, zorder=1)
            prev_was_inf = False
            prev_s = curr_s
            prev_val = curr_val
        else:
            # Show vertical rise when transitioning from finite to infinite
            if not prev_was_inf and prev_s is not None and prev_val is not None:
                ax.plot([prev_s, prev_s], [prev_val, y_max], 'orange',
                       linewidth=1, alpha=0.5, zorder=1)
            prev_was_inf = True

    segments_vel = []
    current_segment_s = []
    current_segment_v = []
    for i, val in enumerate(s_dot_max_vel):
        if val is not None:
            current_segment_s.append(float(s_for_limits[i]))
            current_segment_v.append(float(val))
        else:
            if current_segment_s:
                segments_vel.append((current_segment_s, current_segment_v))
                current_segment_s = []
                current_segment_v = []
    if current_segment_s:
        segments_vel.append((current_segment_s, current_segment_v))

    for i, (seg_s, seg_v) in enumerate(segments_vel):
        label = 'Velocity Limit' if i == 0 else None
        ax.plot(seg_s, seg_v, 'orange', linewidth=1.5, label=label, zorder=2)

    # For failed trajectories: extend limit curve lines through the gap beyond the last
    # integration point. Dashed to distinguish from the confirmed region.
    if lcs_gap_s:
        def build_gap_segments(v_vals):
            segs, curr_s, curr_v = [], [], []
            for s_val, v in zip(lcs_gap_s, v_vals):
                if v is not None:
                    curr_s.append(s_val)
                    curr_v.append(float(v))
                elif curr_s:
                    segs.append((curr_s, curr_v))
                    curr_s, curr_v = [], []
            if curr_s:
                segs.append((curr_s, curr_v))
            return segs

        for seg_s, seg_v in build_gap_segments(lcs_gap_acc):
            ax.plot(seg_s, seg_v, 'r--', linewidth=1.5, alpha=0.6, zorder=2)
        for seg_s, seg_v in build_gap_segments(lcs_gap_vel):
            ax.plot(seg_s, seg_v, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

    # Interior switching points (not start/end)
    # Map kind to marker style
    kind_markers = {
        'k_discontinuous_curvature': ('s', 'Curvature Discontinuity'),
        'k_nondifferentiable_extremum': ('D', 'Non-diff Extremum'),
        'k_velocity_escape': ('v', 'Velocity Escape'),
        'k_discontinuous_velocity_limit': ('p', 'Velocity Limit Discontinuity'),
    }

    kind_seen = set()
    for event in data['events']['backward_starts']:
        kind = event['kind']
        # Skip path begin and end
        if kind in ('k_path_begin', 'k_path_end'):
            continue

        marker, label_text = kind_markers.get(kind, ('^', kind))
        label = label_text if kind not in kind_seen else None
        kind_seen.add(kind)

        # Hollow markers with transparency so they don't obscure the graph
        ax.scatter(float(event['s']), float(event['s_dot']),
                  marker=marker, s=SWITCHING_POINT_SIZE, facecolors='none', edgecolors='blue',
                  alpha=SWITCHING_POINT_ALPHA, linewidths=SWITCHING_POINT_LINE_WIDTH,
                  label=label, zorder=4)

    # Pruned points from splices (dashed green)
    pruned_shown = False
    for splice in data['events']['splices']:
        if 'pruned_points' in splice:
            pruned = splice['pruned_points']
            s_pruned = np.array([float(x) for x in pruned['s']])
            s_dot_pruned = np.array([float(x) for x in pruned['s_dot']])
            label = 'Pruned (replaced)' if not pruned_shown else None
            ax.plot(s_pruned, s_dot_pruned, 'g--', alpha=0.4,
                   linewidth=1.5, label=label, zorder=2.5)
            pruned_shown = True

    # Limit hits (black 'x' markers)
    if data['events']['limit_hits']:
        limit_hit_s = [float(event['s']) for event in data['events']['limit_hits']]
        limit_hit_s_dot = [float(event['s_dot']) for event in data['events']['limit_hits']]
        ax.scatter(limit_hit_s, limit_hit_s_dot, marker='x', s=100,
                   color='black', linewidths=2, label='Limit Hits', zorder=5)

    ax.set_xlabel('Arc Length s')
    ax.set_ylabel('Path Velocity s_dot')
    ax.set_title('Phase Plane Trajectory')
    ax.grid(True, alpha=0.3)
    # Set y-axis limits with buffer below and trajectory visibility above
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)


def plot_joint_trajectories(data, ax):
    """Plot joint-space position trajectories over time."""
    time = np.array([float(x) for x in data['integration_points']['time']])
    configs = data['integration_points']['configuration']
    dof = len(configs[0])

    for joint_idx in range(dof):
        positions = np.array([_f(configs[i][joint_idx]) for i in range(len(configs))])
        ax.plot(time, positions, label=f'Joint {joint_idx}', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Joint Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def plot_joint_velocities(data, ax):
    """Plot joint-space velocity trajectories over time with limit markers."""
    time = np.array([float(x) for x in data['integration_points']['time']])
    velocities = data['integration_points']['velocity']
    dof = len(velocities[0])
    max_velocity = data['metadata']['max_velocity']

    # Store lines and their data for marker placement
    lines_data = []
    for joint_idx in range(dof):
        vels = np.array([_f(velocities[i][joint_idx]) for i in range(len(velocities))])
        line, = ax.plot(time, vels, label=f'Joint {joint_idx}', linewidth=1.5)
        lines_data.append((line, vels, float(max_velocity[joint_idx])))

    # Add limit markers only if trajectory approaches the limit (within 85%)
    for line, vels, limit in lines_data:
        max_vel_reached = np.nanmax(np.abs(vels))
        if max_vel_reached >= 0.85 * limit:
            color = line.get_color()
            ax.plot(0, limit, marker='>', color=color, markersize=LIMIT_MARKER_SIZE,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=10)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Joint Velocities')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def plot_joint_accelerations(data, ax):
    """Plot joint-space acceleration trajectories over time with limit markers."""
    time = np.array([float(x) for x in data['integration_points']['time']])
    accelerations = data['integration_points']['acceleration']
    dof = len(accelerations[0])
    max_acceleration = data['metadata']['max_acceleration']

    # Store lines and their data for marker placement
    lines_data = []
    for joint_idx in range(dof):
        accels = np.array([_f(accelerations[i][joint_idx]) for i in range(len(accelerations))])
        line, = ax.plot(time, accels, label=f'Joint {joint_idx}', linewidth=1.5)
        lines_data.append((line, accels, float(max_acceleration[joint_idx])))

    # Add limit markers only if trajectory approaches the limit (within 85%)
    for line, accels, limit in lines_data:
        max_accel_reached = np.nanmax(np.abs(accels))
        if max_accel_reached >= 0.85 * limit:
            color = line.get_color()
            ax.plot(0, limit, marker='>', color=color, markersize=LIMIT_MARKER_SIZE,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=10)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (rad/s²)')
    ax.set_title('Joint Accelerations')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def plot_arc_length_vs_time(data, ax):
    """Plot arc length progression over time."""
    time = np.array([float(x) for x in data['integration_points']['time']])
    s = np.array([float(x) for x in data['integration_points']['s']])

    ax.plot(time, s, 'b-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Arc Length s')
    ax.set_title('Arc Length vs Time')
    ax.grid(True, alpha=0.3)


def plot_arc_acceleration(data, ax):
    """Plot arc acceleration (s_ddot) vs arc length with feasible bounds."""
    s = np.array([float(x) for x in data['integration_points']['s']])
    s_ddot = np.array([_f(x) for x in data['integration_points']['s_ddot']])

    ip = data['integration_points']
    if 's_ddot_min' in ip and 's_ddot_max' in ip:
        s_ddot_min = np.array([_f(x) for x in ip['s_ddot_min']])
        s_ddot_max = np.array([_f(x) for x in ip['s_ddot_max']])
        ax.fill_between(s, s_ddot_min, s_ddot_max, alpha=0.15, color='red', label='Feasible bounds')
        ax.plot(s, s_ddot_min, 'r-', linewidth=0.5, alpha=0.5)
        ax.plot(s, s_ddot_max, 'r-', linewidth=0.5, alpha=0.5)

    ax.plot(s, s_ddot, 'b-', linewidth=1, label='s_ddot')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Arc Length s')
    ax.set_ylabel('Arc Acceleration s_ddot')
    ax.set_title('Arc Acceleration vs Arc Length')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def main():
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        title = os.path.basename(filename)
        try:
            with open(filename) as f:
                data = load_trajectory(f)
        except FileNotFoundError:
            print(f"Error: File not found: {filename}", file=sys.stderr)
            sys.exit(1)
    else:
        title = "stdin"
        data = load_trajectory(sys.stdin)

    # Create figure with 3-row layout (taller to give phase plane more vertical space)
    fig = plt.figure(figsize=(14, 22), num=title)
    gs = fig.add_gridspec(4, 2)

    # Row 1: Phase plane spans both columns
    ax_phase = fig.add_subplot(gs[0, :])
    plot_phase_plane(data, ax_phase)

    # Row 2: Velocities and accelerations (both have limit markers)
    ax_vel = fig.add_subplot(gs[1, 0])
    plot_joint_velocities(data, ax_vel)

    ax_accel = fig.add_subplot(gs[1, 1])
    plot_joint_accelerations(data, ax_accel)

    # Row 3: Positions and arc length
    ax_pos = fig.add_subplot(gs[2, 0])
    plot_joint_trajectories(data, ax_pos)

    ax_arc = fig.add_subplot(gs[2, 1])
    plot_arc_length_vs_time(data, ax_arc)

    # Row 4: Arc acceleration spans both columns (same x-axis as phase plane)
    ax_s_ddot = fig.add_subplot(gs[3, :])
    plot_arc_acceleration(data, ax_s_ddot)

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for legend at bottom
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
