# trajex

Time-optimal trajectory generation for path following with bounded
acceleration and velocity.

## Overview

trajex is a modern C++20 implementation of the time-optimal path
parameterization algorithm described in:

> Tobias Kunz and Mike Stilman. "Time-Optimal Trajectory Generation for Path Following with Bounded Acceleration and Velocity." Robotics: Science and Systems VIII, 2012.
> https://www.roboticsproceedings.org/rss08/p27.pdf

Given a geometric path in joint space and constraints on joint
velocities and accelerations, trajex computes the time-optimal
trajectory that exactly follows the path while respecting all
constraints.

## References

- Original paper: https://www.roboticsproceedings.org/rss08/p27.pdf
- Reference implementation: https://github.com/tobiaskunz/trajectories

The above papers are also checked into this repo at
- totg/doc/Time-Optimal-Trajectory-Generation.pdf
- totg/doc/Time-Optimal-Trajectory-Generation-Revised.pdf

## Deviations from Paper

Our implementation includes corrections for some errors in the
papers. Some of these errors are corrected in the `-Revised` paper
linked (and embedded) above, but there remain some typos even in that
paper. These are noted in this README as numbered `Correction`s, and
will be similarly identified in the source code.

There are also some algorithmic omissions from the paper, which we
have included in our implementation, along with a few opt-in
improvements. These are noted in this README as numbered `Divergent
Behavior`s, and will be similarly identified in the source code.

Finally, some parts of the paper's algorithm sketch presume implicit
requirements that the paper itself does not articulate, and that the
reference implementation overlooks. Where we have added machinery to
satisfy these requirements, we identify it as a numbered
`Elaboration`, similarly tagged in the source code.

### Corrections to the Papers:

- **Correction 1: Eqs. 7-9**: The `s` in the numerator of the quantity
  passed to the trigonometric functions is incorrect: it should
  instead read `s - s_i`, since what is desired here is the offset
  from the beginning of the circular segment, not the absolute arc
  length.

- **Correction 2: VI.3**: As noted in the `-Revised` paper above, the
  RHS of the inequalities should say `vel`, not `acc`.

- **Correction 3: Eq. 38**: Two of the expressions are missing "dots",
  since there is no `max_acc` for `s`, only for `s_dot`.

- **Correction 4: Eq 38**: The equation is not dimensionally sound,
  since the result of the `s_dot_dot_max` function is an acceleration
  term, but it is compared to a slope in the phase plane. Instead, we
  interpret these equations more like VI.3, where there is an `s_dot`
  in the denominator, which renders it dimensionally
  sound. Practically, we divide the LHS by `s_dot_max_acc(s+-)` when
  comparing in the inequality against the slope of
  `s_dot_max_acc(s+-)`

- **Correction 5: Eq. 40**: As noted in the `-Revised` paper above, on
  the LHS the third `s` should be dotted, and the RHS should say
  `vel`, not `acc`. However, additionally, the LHS needs to be divided
  by an `s_dot`-type quantity, much like in `Correction 4`.

- **Correction 6: Eqs. 41 and 42**: The third `s` in the LHS of each
  inequality is missing a dot. Furthermore, similar to the above two
  corrections, the LHS needs to be divided by the appropriate
  `s_dot`-type quantity.

- **Correction 7: Eq. 38**: In the positive step case, we are looking
  for a *sink*, which would mean all candidate accelerations are
  infeasible, so we check against the minimum, not the maximum.

- **Correction 8: Eqs. 41 and 42**: The RHS of both inequalities
  references `d/ds s_dot_max_acc`, but should reference `d/ds
  s_dot_max_vel`. Equations 41 and 42 are the discontinuous analog of
  equation 40, which compares the min-acceleration trajectory slope
  against the velocity limit curve slope to determine whether the
  trajectory can follow below the velocity limit curve. The same
  comparison applies at a discontinuity: on the before-side, check if
  the velocity limit is a sink; on the after-side, check if it is a
  source or followable. Both checks compare against the velocity limit
  curve slope, not the acceleration limit curve slope. This is
  consistent with equation 40 and with the analogous pattern in
  equation 38, where sink/source checks at the acceleration limit curve
  compare against the acceleration limit curve slope.

### Behavioral Differences:

- **Divergent Behavior 1**: We implement an opt-in denoising pass for
  input waypoints. If a sequence of waypoints can be contained within
  a forward extending cylinder with a user specified diameter, those
  points are removed and no circular blend is produced for them. See
  `path::options::max_linear_deviation` for more details.

- **Divergent Behavior 2**: The backward integration pass
  conservatively rejects trajectories that exceed limit curves, which
  is not described in the paper.

- **Divergent Behavior 3**: Blend curvature is bounded between
  `min_blend_curvature` and `max_blend_curvature` (see
  `path::options`). Near-collinear waypoints produce enormous blend
  radii causing catastrophic cancellation in arc arithmetic; we cap
  the radius at `1/min_blend_curvature` while retaining exact C1
  continuity. Near-reversal waypoints produce degenerate tiny arcs;
  above `max_blend_curvature` we emit an unblended L-L corner instead,
  handled by Divergent Behavior 4.

- **Divergent Behavior 4**: The paper assumes all segment boundaries
  are C1 tangent-continuous by construction. We check the tangent dot
  product at every segment boundary during forward integration and
  switching-point search. Any boundary with a tangent discontinuity —
  including the L-L corners produced by Divergent Behavior 3 —
  mandates `s_dot = 0`. The reference implementation does not perform
  this check.

- **Divergent Behavior 5**: The paper assumes L-C-L topology where
  circular blends are always separated by linear segments. When two
  adjacent blends fully consume the connecting segment, we allow
  directly adjacent circular arcs (C-C). C-C boundaries are
  tangent-continuous by blend construction, so Divergent Behavior 4
  passes through without stopping. The Case 2 switching-point search
  is extended to handle extrema at C-C boundaries via limit-curve
  continuity checking across the boundary.

### Elaborations:

- **Elaboration 1: Backward integration solve**: The paper's
  "integrate backward with minimum acceleration" in VI Step 5
  implicitly requires consecutive backward points to be connected by a
  forward step that uses exactly `s_ddot_min` -- the most negative
  path acceleration the joint acceleration constraints permit. On
  curved segments the path tangent and curvature vary continuously
  with `s`, so `s_ddot_min` varies with `s` as well, and the local
  `s_ddot_min` at one backward point differs from the one a forward
  replay would see at the next; the implicit consistency breaks at
  every step. (Linear segments avoid this entirely because
  `s_ddot_min` is constant in `s` along them.) The reference
  implementation steps with the local `s_ddot_min` and accepts the
  inconsistency, producing trajectories with acceleration-bound
  excursions at backward-stepped points. We solve at each backward
  step for the velocity at which decelerating forward at the local
  `s_ddot_min` lands exactly on the previous backward point,
  eliminating the inconsistency at its source.

## Extensions

- **Extension 1: TCP speed limit**: An optional bound on the Cartesian
  linear speed of the tool center point, enabled by setting
  `trajectory::options::tcp` with a speed cap and a kinematics model, where
  the source carries the matching `Extension 1` tag. When it is unset, none
  of what follows applies and the algorithm is unchanged.

  The constraint reduces to a third limit curve. From `q = f(s)` we have
  `q_dot = f'(s) s_dot`, and the TCP velocity is `v = J_v(q) q_dot`, so
  `||v|| = ||J_v(f(s)) f'(s)|| s_dot`. Writing `g(s)` for that gain and
  bounding `||v||` by the cap `v_max` gives `s_dot_max_tcp(s) = v_max /
  g(s)`. Where `g(s)` falls below `epsilon` the TCP is stationary however
  fast we traverse the path, so the curve is infinite there and cannot
  constrain; a NaN Jacobian is treated the same way.

  This is the same kind of object as the joint velocity limit curve of
  Eq. 36, being the `s_dot` at which a quantity linear in `s_dot`
  saturates. Section VII already composes the acceleration and velocity
  limit curves by taking whichever is lower, so we compose this one the
  same way. The algorithm therefore sees a single combined velocity limit
  curve, rather than a third curve to be searched and followed separately.

  Taking a minimum introduces kinks where the two components cross, and
  the switching point machinery has to survive them. It does, because such
  a kink is always concave: at a crossing the newly active curve is the one
  descending through the other, so the slope of the combined curve drops.
  The Eq. 40 escape quantity, `s_ddot_min / s_dot` less the curve slope,
  can therefore only jump upward at a crossing, and the continuous search
  triggers only on a transition from positive to non-positive, so a
  crossing cannot be mistaken for a switching point. The opposite case,
  where the combined curve begins falling faster than maximum braking can
  follow, is an entry into a trapped region, which the trap check during
  curve following catches; the next genuine switching point is then found
  downstream and backward integration sweeps back through the kink. This is
  the same argument that already covers the kinks the joint curve produces
  on its own when the limiting joint changes, so TCP supplies more kinks
  rather than a new problem.

  What that argument does not cover is a miss. The continuous switching
  point search samples the escape quantity in fixed strides and resets its
  baseline at segment boundaries, since a sign change across a geometric
  discontinuity is not a continuous escape, but it does not know about
  crossings, which fall in a segment's interior. A stride containing both a genuine root and
  a crossing beyond it reads positive at both ends, because the same upward
  jump that prevents false positives also hides the root, and no bracket
  forms. The consequence is bounded: we follow the curve longer than
  necessary, or pick up a switching point further downstream, and never
  violate a limit. Kunz & Stilman acknowledge the same class of miss for
  their own numerical search in Section VIII-B, and the joint curve's own
  kinks already expose the algorithm to it.

  The slope of the combined curve comes from whichever component is active,
  and the comparison deciding which is exact rather than epsilon-wrapped.
  Near a crossing the two curves sit within any epsilon of each other while
  their slopes differ arbitrarily, so widening that comparison would let it
  return the slope of the curve that is not binding. Ties go to the joint
  curve, matching the `min()` that produced the value, so the value and the
  slope never come from different curves.

  Finally, note what the cap bounds. It bounds the TCP speed of the
  kinematics model it was given, and nothing at runtime compares the result
  against a measured Cartesian speed, so a stale or incorrect model table
  produces a confidently wrong limit.
