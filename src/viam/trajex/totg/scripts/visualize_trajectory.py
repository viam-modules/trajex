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


def _plot_segmented(ax, s_list, v_list, *, label=None, **plot_kwargs):
    """Plot a limit curve that may contain None (infinite) entries as separate finite
    segments. The label is applied only to the first segment so the legend gets one entry."""
    seg_s, seg_v = [], []
    first = True
    for s_val, v in zip(s_list, v_list):
        if v is not None:
            seg_s.append(float(s_val))
            seg_v.append(float(v))
        elif seg_s:
            ax.plot(seg_s, seg_v, label=(label if first else None), **plot_kwargs)
            first = False
            seg_s, seg_v = [], []
    if seg_s:
        ax.plot(seg_s, seg_v, label=(label if first else None), **plot_kwargs)


def _plot_inf_transitions(ax, s_list, v_list, y_max, **plot_kwargs):
    """Draw vertical indicator lines where a limit curve transitions between finite and
    infinite (None) entries, so a curve leaving the top of the plot reads differently
    from a gap in the data."""
    prev_was_inf = True
    first_entry = True
    prev_s = None
    prev_val = None
    for s_val, v in zip(s_list, v_list):
        curr_s = float(s_val)
        if v is not None:
            curr_val = float(v)
            # Vertical drop when transitioning from infinite to finite
            if prev_was_inf and not first_entry:
                ax.plot([curr_s, curr_s], [y_max, curr_val], **plot_kwargs)
            prev_was_inf = False
            prev_s = curr_s
            prev_val = curr_val
        else:
            # Vertical rise when transitioning from finite to infinite
            if not prev_was_inf:
                ax.plot([prev_s, prev_s], [prev_val, y_max], **plot_kwargs)
            prev_was_inf = True
        first_entry = False


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

    # Component velocity limits (joint-only and TCP-only), when the serializer provides them.
    # These let us draw the joint and TCP curves separately so their crossing is visible.
    ip = data['integration_points']
    s_ip = [float(x) for x in ip['s']]
    joint_vel = ip.get('s_dot_max_vel_joint')
    tcp_vel = ip.get('s_dot_max_vel_tcp')
    has_tcp_components = joint_vel is not None

    lcs = data.get('limit_curve_samples')
    lcs_gap_s, lcs_gap_acc, lcs_gap_vel = [], [], []
    lcs_gap_joint, lcs_gap_tcp = [], []

    merged = list(zip(
        s_ip,
        ip['s_dot_max_acc'],
        ip['s_dot_max_vel'],
        joint_vel if joint_vel is not None else [None] * len(s_ip),
        tcp_vel if tcp_vel is not None else [None] * len(s_ip),
    ))
    if lcs:
        # Joint/TCP component samples are present only on dumps that recorded them; older dumps
        # carry just the combined curve. Fall back to all-None so the gap overlay degrades cleanly.
        lcs_joint = lcs.get('s_dot_max_vel_joint', [None] * len(lcs['s']))
        lcs_tcp = lcs.get('s_dot_max_vel_tcp', [None] * len(lcs['s']))
        for s_val, acc_val, vel_val, joint_val, tcp_val in zip(
                lcs['s'], lcs['s_dot_max_acc'], lcs['s_dot_max_vel'], lcs_joint, lcs_tcp):
            s_f = float(s_val)
            if s_f <= last_s:
                merged.append((s_f, acc_val, vel_val, joint_val, tcp_val))
            else:
                lcs_gap_s.append(s_f)
                lcs_gap_acc.append(acc_val)
                lcs_gap_vel.append(vel_val)
                lcs_gap_joint.append(joint_val)
                lcs_gap_tcp.append(tcp_val)

    merged.sort(key=lambda x: x[0])
    s_for_limits = [x[0] for x in merged]
    s_dot_max_acc = [x[1] for x in merged]
    s_dot_max_vel = [x[2] for x in merged]
    joint_for_limits = [x[3] for x in merged]
    tcp_for_limits = [x[4] for x in merged]

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
    if has_tcp_components:
        for series in (joint_vel, tcp_vel):
            for val in (series or []):
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
    _plot_inf_transitions(ax, s_for_limits, s_dot_max_acc, y_max,
                          color='red', linewidth=1, alpha=0.5, zorder=1)
    _plot_segmented(ax, s_for_limits, s_dot_max_acc, color='red', linewidth=1.5,
                    label='Acceleration Limit', zorder=2)

    # Velocity limit curve(s). When the serializer provides the joint and TCP components
    # separately (TCP-limited runs), plot them as distinct curves so their crossing is visible,
    # plus a thin overlay of the combined min(joint, TCP) curve the integrator actually rides.
    # Otherwise plot the single combined curve. All curves use the merged sample set so any
    # denser limit_curve_samples in range (e.g. around pruned points) are included.
    if has_tcp_components:
        _plot_inf_transitions(ax, s_for_limits, joint_for_limits, y_max,
                              color='orange', linewidth=1, alpha=0.5, zorder=1)
        _plot_segmented(ax, s_for_limits, joint_for_limits, color='orange', linewidth=1.5,
                        label='Joint Velocity Limit', zorder=2)
        if any(v is not None for v in tcp_for_limits):
            _plot_inf_transitions(ax, s_for_limits, tcp_for_limits, y_max,
                                  color='purple', linewidth=1, alpha=0.5, zorder=1)
            _plot_segmented(ax, s_for_limits, tcp_for_limits, color='purple', linewidth=1.5,
                            label='TCP Velocity Limit', zorder=2)
        _plot_segmented(ax, s_for_limits, s_dot_max_vel, color='dimgray', linewidth=1.0,
                        linestyle=':', label='Velocity Limit (min)', zorder=2.5)
    else:
        _plot_inf_transitions(ax, s_for_limits, s_dot_max_vel, y_max,
                              color='orange', linewidth=1, alpha=0.5, zorder=1)
        _plot_segmented(ax, s_for_limits, s_dot_max_vel, color='orange', linewidth=1.5,
                        label='Velocity Limit', zorder=2)

    # For failed trajectories: extend limit curve lines through the gap beyond the last
    # integration point. Dashed to distinguish from the confirmed region.
    if lcs_gap_s:
        _plot_segmented(ax, lcs_gap_s, lcs_gap_acc, color='red', linestyle='--',
                        linewidth=1.5, alpha=0.6, zorder=2)

        # Mirror the integration-point region: when the joint/TCP split is available, extend both
        # components through the gap (orange joint, purple TCP) instead of only the combined curve.
        gap_has_components = has_tcp_components and any(v is not None for v in lcs_gap_joint)
        if gap_has_components:
            _plot_segmented(ax, lcs_gap_s, lcs_gap_joint, color='orange', linestyle='--',
                            linewidth=1.5, alpha=0.6, zorder=2)
            _plot_segmented(ax, lcs_gap_s, lcs_gap_tcp, color='purple', linestyle='--',
                            linewidth=1.5, alpha=0.6, zorder=2)
        else:
            _plot_segmented(ax, lcs_gap_s, lcs_gap_vel, color='orange', linestyle='--',
                            linewidth=1.5, alpha=0.6, zorder=2)

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

    # Allow the caller to label the window (e.g. when the JSON arrives on stdin).
    title = os.environ.get("GRAPH_TITLE", title)

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
