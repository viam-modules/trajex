#!/usr/bin/env python3
"""
Plot the TCP (tool center point) Cartesian speed over time from a trajectory
visualization JSON read on stdin (the output of viam-trajex-totg-replay).

The TCP speed is derived from the serialized phase-plane data rather than
recomputed from kinematics. The TCP limit curve is

    s_dot_max_vel_tcp(s) = tcp_cap / ||J(q) * f'(s)||

and the actual TCP speed is ||J(q) * f'(s)|| * s_dot, so

    tcp_speed(t) = tcp_cap * s_dot(t) / s_dot_max_vel_tcp(t).

Where the TCP limit is infinite (||J * f'|| approx 0, the TCP is momentarily
stationary while the joints move) the TCP speed is 0.

The cap is read from the TCP_SPEED_CAP environment variable (m/s). Without it,
the plot shows the speed as a fraction of the cap (same shape, unscaled).
GRAPH_TITLE sets the window title.
"""

import json
import os
import sys

import matplotlib.pyplot as plt


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        sys.exit(f"error: invalid JSON on stdin: {e}")

    ip = data.get("integration_points")
    if not ip or not ip.get("time"):
        sys.exit("error: no integration_points in input (failed or empty trajectory?)")

    tcp_lim = ip.get("s_dot_max_vel_tcp")
    if tcp_lim is None:
        sys.exit("error: input has no s_dot_max_vel_tcp; this record has no TCP limit")
    # A joint-only move still carries the key, but every entry is null (the TCP limit is
    # infinite everywhere). There is no TCP speed to plot in that case.
    if all(v is None for v in tcp_lim):
        sys.exit("error: this record has no TCP limit (s_dot_max_vel_tcp is infinite throughout); "
                 "nothing to plot")

    time = [float(x) for x in ip["time"]]
    s_dot = [float(x) for x in ip["s_dot"]]

    cap_env = os.environ.get("TCP_SPEED_CAP", "").strip()
    try:
        cap = float(cap_env) if cap_env else None
    except ValueError:
        sys.exit(f"error: TCP_SPEED_CAP must be a number in m/s, got {cap_env!r}")

    speed = []
    for sd, lim in zip(s_dot, tcp_lim):
        if lim is None:            # TCP limit infinite => ||J*f'|| approx 0 => TCP stationary
            ratio = 0.0
        elif float(lim) <= 0.0:    # degenerate; avoid divide-by-zero
            ratio = float("nan")
        else:
            ratio = sd / float(lim)
        speed.append(cap * ratio if cap is not None else ratio)

    title = os.environ.get("GRAPH_TITLE", "TCP speed")
    fig, ax = plt.subplots(figsize=(14, 6), num=title)
    ax.plot(time, speed, color="purple", linewidth=1.5, label="TCP speed", zorder=3)

    if cap is not None:
        ax.axhline(cap, color="red", linestyle="--", linewidth=1, label=f"TCP cap ({cap:g} m/s)")
        ax.set_ylabel("TCP speed (m/s)")
    else:
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="cap")
        ax.set_ylabel("TCP speed (fraction of cap)")

    ax.set_xlabel("Time (s)")
    ax.set_title("TCP Cartesian Speed vs Time")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
