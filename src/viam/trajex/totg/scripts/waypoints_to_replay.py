#!/usr/bin/env python3
"""
Convert a raw joint-space waypoint dump into a canonical TOTG replay record.

Input is a JSON array of arrays: N rows, each row a DOF-length list of joint
positions in radians. The output conforms to the schema consumed by
viam::trajex::totg::replay_planner::create (see src/viam/trajex/totg/tools/replay.cpp).

The converter does NOT deduplicate waypoints. Duplicate-row handling is left to
the consumer of the replay record (e.g., the test that drives a streaming
session).

Velocity, acceleration, and path-tolerance limits are accepted on the command
line in degree units for readability and converted to radians internally to
match the schema.

Typical usage:

    waypoints_to_replay.py \\
        --max-vel-deg-per-sec 20 \\
        --max-accel-deg-per-sec2 20 \\
        --output path/to/output.trajex-totg-replay.json \\
        path/to/raw_waypoints.json

Streams default to stdin/stdout when the positional input or `--output` is
omitted (or set to '-'), enabling pipeline usage:

    cat raw.json | waypoints_to_replay.py --max-vel-deg-per-sec 20 \\
        --max-accel-deg-per-sec2 20 > out.json
"""

import argparse
import datetime
import json
import math
import sys
from pathlib import Path


SCHEMA_VERSION = 1

# The converter intentionally holds no opinions on the optional trajex
# parameters (blend curvature bounds, colinearization ratio, path tolerance).
# When a CLI flag is not provided, the corresponding JSON field is omitted from
# the record, and the C++ replay deserializer falls back to whichever trajex
# default applies (path::options defaults for the curvature bounds, nullopt for
# colinearization_ratio, and planner_base::config's 0.0 for path tolerance).
# Use of this tool with incomplete overrides is the user's responsibility: the
# point is to merge waypoints with externally-supplied parameters, not to guess
# the parameters.


def load_waypoints(stream, source_label):
    """Read a raw waypoint dump from `stream` and return (waypoints, dof).

    `source_label` is used purely for error messages so callers can convey the
    origin (a path or "<stdin>") without the loader caring which.

    Validates that the input is a non-empty rectangular array of arrays with at
    least two rows. Raises ValueError with a descriptive message on any
    structural problem.
    """
    data = json.load(stream)

    if not isinstance(data, list) or not data:
        raise ValueError(f"{source_label}: expected a non-empty JSON array of waypoints")
    if len(data) < 2:
        raise ValueError(f"{source_label}: at least two waypoints required, got {len(data)}")

    dof = None
    for i, row in enumerate(data):
        if not isinstance(row, list):
            raise ValueError(f"{source_label}: waypoint {i} is not an array")
        if dof is None:
            dof = len(row)
            if dof == 0:
                raise ValueError(f"{source_label}: waypoint 0 has zero joints")
        elif len(row) != dof:
            raise ValueError(
                f"{source_label}: waypoint {i} has {len(row)} joints, "
                f"expected {dof} (matching waypoint 0)"
            )
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)):
                raise ValueError(f"{source_label}: waypoint {i}, joint {j} is not numeric")

    return data, dof


def build_record(waypoints, dof, args):
    """Assemble the replay record as a dict matching the canonical schema.

    Only the mandatory fields (vel/accel limits, waypoints) and the optional
    fields the user explicitly supplied on the command line are emitted.
    Omission is meaningful: the deserializer treats absent optional fields as
    "use the trajex default" rather than "default to whatever the script picks."
    """
    deg_to_rad = math.pi / 180.0
    vel_rad_per_sec = args.max_vel_deg_per_sec * deg_to_rad
    accel_rad_per_sec2 = args.max_accel_deg_per_sec2 * deg_to_rad

    # The replay deserializer keys on field names; ordering does not matter for
    # parsing. json.dumps(sort_keys=True) downstream renders the file with the
    # same alphabetical ordering used by the C++ jsoncpp StyledWriter for the
    # existing replay records, so converted files visually match precedent.
    record = {
        "max_acceleration_vec_rads_per_sec2": [accel_rad_per_sec2] * dof,
        "max_velocity_vec_rads_per_sec": [vel_rad_per_sec] * dof,
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "waypoints_rads": waypoints,
    }
    if args.max_blend_curvature is not None:
        record["max_blend_curvature"] = args.max_blend_curvature
    if args.min_blend_curvature is not None:
        record["min_blend_curvature"] = args.min_blend_curvature
    if args.path_colinearization_ratio is not None:
        record["path_colinearization_ratio"] = args.path_colinearization_ratio
    if args.path_tolerance_deg is not None:
        record["path_tolerance_delta_rads"] = args.path_tolerance_deg * deg_to_rad
    return record


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help=(
            "Path to the raw waypoint dump (JSON array of arrays, radians). "
            "Use '-' or omit to read from stdin."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help=(
            "Path to write the replay record to. Use '-' or omit to write to stdout."
        ),
    )
    parser.add_argument(
        "--max-vel-deg-per-sec",
        type=float,
        required=True,
        help="Per-joint velocity limit, applied uniformly across all DOF.",
    )
    parser.add_argument(
        "--max-accel-deg-per-sec2",
        type=float,
        required=True,
        help="Per-joint acceleration limit, applied uniformly across all DOF.",
    )
    parser.add_argument(
        "--max-blend-curvature",
        type=float,
        default=None,
        help="Upper bound on blend curvature. Omitted from output if unset.",
    )
    parser.add_argument(
        "--min-blend-curvature",
        type=float,
        default=None,
        help="Lower bound on blend curvature. Omitted from output if unset.",
    )
    parser.add_argument(
        "--path-colinearization-ratio",
        type=float,
        default=None,
        help="Path colinearization ratio. Omitted from output if unset.",
    )
    parser.add_argument(
        "--path-tolerance-deg",
        type=float,
        default=None,
        help="Path blend tolerance, in degrees. Omitted from output if unset.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        if args.input == "-":
            waypoints, dof = load_waypoints(sys.stdin, "<stdin>")
        else:
            input_path = Path(args.input)
            with open(input_path, "r") as f:
                waypoints, dof = load_waypoints(f, str(input_path))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    record = build_record(waypoints, dof, args)

    if args.output == "-":
        json.dump(record, sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(record, f, indent=1, sort_keys=True)
            f.write("\n")
        print(
            f"wrote {output_path} (dof={dof}, waypoints={len(waypoints)})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
