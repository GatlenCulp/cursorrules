#import "preamble.typ": *


#show: cheatsheet.with(
  title: "Midterm",
  date: datetime(year: 2025, month: 11, day: 5),
)

#let gatbox = gatbox.with(color-cycle: true)
#let scal(x) = $#x$

#gatbox(
  title: [#fa-icon("hand") Ch. 01: Introduction],
)[
  _High-level goals, scope, and examples_

  == #fa-icon("lightbulb") Overview

  - *Core challenges:* Contact-rich dynamics; integrate perception, planning, and control.
  - *Beyond pick-and-place:* Task diversity; suction common but insufficient for many tasks.
  - *Tools today:* Strong rendering and contact simulation; runnable notebooks for practice.
  - *Systems view:* Components/diagrams; ROS message contracts vs Drake state/timing semantics; visualize system graphs for profiling.
]

#gatbox(
  title: [#fa-icon("file-code") Ch. 02: Robot Setup],
)[
  _Standard models enable reuse across robots_

  == #fa-icon("layer-group") Essentials

  - *Formats:* URDF/SDF (primary), limited MJCF; Drake Model Directives (YAML) for multi-robot composition/edits.
  - *Best practices:* One source of truth for kinematics/inertias/visuals/collisions; validate in visualizer.
  - *Frames:* Clear base/eef/camera frames simplify IK and calibration.

  == #fa-icon("ruler-combined") Modeling Details

  *Inertias:* Use consistent units; verify link mass m, center c, and inertia matrix about c are physical (symmetric, positive definite).

  *Collision vs visual:* Keep collision simple/convex where possible; ensure no self-collisions in nominal poses.

  *Limits:* Encode $q in [q_"min", q_"max"]$, $abs(dot(q)) <= dot(q)_"max"$, $abs(tau) <= tau_"max"$; validate in the visualizer.

  *Self-checks:* Spawn gravity-only sim → verify $tau_g (q)$ balancing; drop tests for contacts and restitution.

  _Control interfaces and why torque sensing matters_

  == #fa-icon("bullseye") Position-Controlled Robots

  *Definition:* Track joint positions/trajectories precisely; typical when torque interface is unavailable.

  *Why common:* Electric motors with large gear reductions break simple $tau <- k_t i$ due to backlash, friction, and unmodeled transmission dynamics → robust torque control is hard; closed-loop position control is easier.

  == #fa-icon("scale-balanced") Torque-Controlled Robots

  *Capability:* Joint-torque sensing + high-rate torque commands enables compliant behaviors, contact-rich tasks, and force-control.

  *Example platform:* KUKA LBR iiwa used throughout notes for torque control experiments.

  == #fa-icon("sliders") Practical Guidance

  *If you have torque control:* You can still do high-accuracy position control; prefer torque mode when contact/stiffness/compliance matters.

  *If you only have position control:* Use impedance/stiffness at trajectory level, add compliance via end-effector mechanisms, and favor contact-robust strategies.

  == #fa-icon("industry") Transmission & Reflected Dynamics

  *Motor current to torque:* $tau_"motor" := k_t i$ (ideal).

  *With gear ratio $N$ and efficiency $eta$:* $tau_"joint" = eta N tau_"motor" - tau_"fric"$.

  *Reflected load:* $J_"reflected" := N^2 J_"load"$, $b_"reflected" := N^2 b_"load"$; large N amplifies unmodeled dynamics (backlash, friction).

  == #fa-icon("wave-square") Impedance over Position Interface

  *Joint-space impedance:* $tau_"ref" := K_p (q_"des" - q) + K_d (dot(q)_"des" - dot(q)) + K_i E_q + tau_"ff"$ where $E_q$ accumulates error. Position-only APIs approximate this by shaping commanded trajectories and internal gains.

  *Cartesian impedance:* $f := K_x (x_"des" - x) + D_x (dot(x)_"des" - dot(x))$, $tau_"cmd" <- J^T f + N^T tau_"null"$.

  == #fa-icon("gauge") Torque-Control Architecture

  *Gravity compensation:* $tau_g (q)$ added to improve passivity and reduce effort.

  *Full command:* $tau_"cmd" <- tau_g (q) + tau_"impedance" + tau_"ff"$; respect $norm(tau_"cmd")_oo <= tau_"max"$ and rate limits.

  _Trade-offs among dexterous, simple, and special-purpose grippers_

  == #fa-icon("fingerprint") Dexterous Hands

  *Pros:* In-hand manipulation, rich contact modalities, versatile.

  *Cons:* Complex control, sensing, calibration; lower reliability in clutter; higher cost.

  == #fa-icon("hand-back-fist") Simple/Underactuated Grippers

  *Pros:* Robust, cheap, tolerant to pose error; emergent adaptability (underactuation/compliance).

  *Cons:* Limited in-hand reorientation; rely on environment for dexterity.

  == #fa-icon("magnet") Suction and Specialized Tools

  *Suction:* Excellent for flat or sealed surfaces; struggles on porous/rough geometry; add sensors for seal detection.

  *Tooling:* Choose end-effectors per task physics (pinch, power grasp, hooks, spatulas) and expected object set.

  == #fa-icon("hand-holding") Contact Modeling Notes

  *Friction:* Use Coulomb cone approximations in planners; account for stick/slide transitions in controllers.

  *Suction:* Seal depends on surface curvature/roughness; add vacuum sensing; treat payload as $w := m g$ with margin for accelerations.

  *Compliance:* Underactuation/compliant pads widen successful grasp set but reduce precise in-hand dexterity.

  _Perception and proprioception for manipulation_

  == #fa-icon("camera") Exteroception

  *RGB/Depth/Cameras:* Pose estimation, scene understanding; mount on wrist and/or fixed viewpoints; mind occlusions and calibration.

  *Tactile/Proximity:* Detect incipient contact, slip, and shear; improves grasp reliability and insertion tasks.

  == #fa-icon("microchip") Proprioception

  *Joint encoders & velocities:* Core for control/estimation.

  *Joint torque/force sensors:* Enable model-based force control and safe contact.

  *FT sensor at wrist:* Measures interaction wrench; useful for hybrid position/force tasks.

  == #fa-icon("compass-drafting") Calibration & Noise

  *Hand–eye (eye-in-hand):* Solve $A_i X = X B_i$ for camera–eef transform X from pairs of robot motions $(A_i)$ and observed camera motions $(B_i)$.

  *Noise models:* $z := h(x) + v$, $v :in cal(N) (0, Sigma)$; propagate to pose and contact estimation; filter with complementary/EKF as needed.

  *Time sync:* Align sensor timestamps with controller loop to avoid phase lag in feedback.

  _From models to runnable systems_

  == #fa-icon("server") HardwareStation

  *Purpose:* Defines robot(s), sensors, controllers, scene objects, and wiring in one place; consistent interfaces for sim and real.

  *Usage:* Load description files + directives, set controllers (position/torque), expose ports for commands and measurements.

  == #fa-icon("gamepad") Simulation

  *Modern rendering:* Synthetic images can test/train perception with real-world transfer.

  *Contact simulation:* Improved solvers make multi-body contact practical; validate grasp/contact strategies in silico before hardware.

  == #fa-icon("wrench") Workflow Tips

  *Unified config:* Keep station configs shared across sim/real to minimize drift.

  *Safety first:* Rate-limit torques/velocities; test contact behaviors in sim; bring up with high damping/compliance.

  == #fa-icon("share-nodes") Ports & Rates

  *Inputs:* $q_"des"$, $dot(q)_"des"$, $tau_"cmd"$ (mode-dependent). *Outputs:* $q, dot(q), tau, "wrench"$, images, depth.

  *Rates:* Control (1–2 kHz torque, 250–500 Hz position), perception (30–60 Hz RGB, 10–30 Hz depth); buffer and decimate appropriately.

  == #fa-icon("shield") Bring-Up Checklist

  *Soft limits:* Enforce $q, dot(q), tau$ bounds and collision margins.

  *Validation:* Gravity comp on → move slowly → contact probing at low stiffness → task execution with logging.

  _Key facts/settings commonly queried in PS01_

  == #fa-icon("info-circle") IIWA14 Facts

  *Joint count:* 7; *Link count (incl. base & eef):* 8; *Joint type:* revolute; *Control:* torque.

  == #fa-icon("sliders") Position Controller Setup

  *Gain:* $K_p approx 100$–$200$ typical for settling (PS01 uses $120$). Tune $K_d$/$K_i$ as needed, or omit.

  *Command:* Set $q_"des"$; controller computes $tau_"ref"$; plant applies $tau_"cmd"$ after gravity/limits.

  *Initial conditions:* Example $q_0 := [0.2, 0.2, 0.2, 0, 0, 0, 0]$; $q_"des" := [0, 0, 0, 0, 0, 0, 0]$; simulate for $T := 10$ s, then read final $q(T)$.

  == #fa-icon("project-diagram") Systems & Drake Features

  *Block-diagram systems*, *optimization toolkit*, *multi-body dynamics*, *deterministic replay*; not GPU-parallel by default.

  == #fa-icon("server") HardwareStation Playback

  *Usage:* Launch station (sim or real config), wire controllers/sensors, run for horizon $T$, log ports; export a screen recording for verification tasks.
]

#gatbox(
  title: [#fa-icon("diagram-project") Ch. 03: Kinematics & Pick-and-Place],
)[
  _Frames, positions, rotations, transforms_

  == #fa-icon("location-dot") Frames, Points, Positions

  *Positions:* Use monogram with attach.
  $ attach(p, tl: A, tr: C)_(F) := "pos of C measured from A, expressed in F" $.
  Shorthands: if expressed-in equals measured-from, drop subscript. If measured-from is $W$, drop tl.

  *Rotations:* $ attach(R, tl: A, tr: B) := "orient of B measured from A" $.
  Composition/inverse: $attach(R, tl: A, tr: B) attach(R, tl: B, tr: C) = attach(R, tl: A, tr: C)$; $(attach(R, tl: A, tr: B))^(-1) = attach(R, tl: B, tr: A)$.

  *Transforms (poses):* $attach(X, tl: A, tr: B)$ bundles translation+rotation. Position/composition: $attach(p, tl: G, tr: A) = attach(X, tl: G, tr: F) attach(p, tl: F, tr: A)$; $attach(X, tl: A, tr: B) attach(X, tl: B, tr: C) = attach(X, tl: A, tr: C)$.

  == #fa-icon("camera") Camera-to-World Conversion

  If camera frame $C$ has pose $attach(X, tl: W, tr: C)$, a camera point $attach(p, tl: C, tr: P_i)$ maps as $attach(p, tl: W, tr: P_i) = attach(X, tl: W, tr: C) attach(p, tl: C, tr: P_i)$ with inverse extrinsics $attach(X, tl: C, tr: W)$.

  _Kinematic tree, frame composition, representations_

  == #fa-icon("sitemap") Kinematic Tree and Joint Frames

  Each joint defines $attach(X, tl: J_P, tr: J_C) (q)$ and fixed offsets $attach(X, tl: P, tr: J_P), attach(X, tl: J_C, tr: C)$.
  Between parent $P$ and child $C$: $attach(X, tl: P, tr: C) (q) = attach(X, tl: P, tr: J_P) attach(X, tl: J_P, tr: J_C) (q) attach(X, tl: J_C, tr: C)$.

  Goal (gripper pose): $X^G := f_"kin"^G (q)$ via recursive composition.

  == #fa-icon("rotate") 3D Rotation Representations

  Rotation matrices, roll-pitch-yaw (RPY), axis-angle, and unit quaternions have trade-offs (RPY gimbal lock at pitch = pi/2). Use quaternions for representation; use angular velocity for derivatives.

  == #fa-icon("right-left") Generalized Velocities

  Do not assume $dot(q) = v$. Floating-base often uses quaternions in $q$ and angular velocities in $v$. Drake: `MapQDotToVelocity`, `MapVelocityToQDot`.

  _Spatial velocity, geometric vs analytic Jacobians_

  == #fa-icon("vector-square") Spatial Velocity

  6D twist: $attach(V, tl: A, tr: B)_(C) := [ attach(omega, tl: A, tr: B)_(C); quad attach(v, tl: A, tr: B)_(C) ] in RR^6$. Change of expressed-in frame: $attach(omega, tl: A, tr: B)_(G) = attach(R, tl: G, tr: F) attach(omega, tl: A, tr: B)_(F)$; $attach(v, tl: A, tr: B)_(G) = attach(R, tl: G, tr: F) attach(v, tl: A, tr: B)_(F)$.

  == #fa-icon("square-root-variable") Jacobian Mapping

  Geometric Jacobian (w.r.t. generalized velocity $v$): $attach(V, tl: W, tr: G) = J^(G) (q) v$.
  Available: `CalcJacobianAngularVelocity`, `CalcJacobianTranslationalVelocity`, `CalcJacobianSpatialVelocity` (choose w.r.t. $dot(q)$ or $v$).

  == #fa-icon("triangle-exclamation") Singularities & Manipulability

  Track $sigma_"min" (J)$; near-zero inflates $(J)^+$. Kinematic singularities when $rank(J) < 6$. Manipulability ellipsoid: image of unit ball in joint-velocity space through $J$.

  _Pseudo-inverse, QP with bounds, joint centering, pose tracking_

  == #fa-icon("equals") Pseudo-inverse as Least Squares

  Unconstrained least-squares: $v^* := arg min_(v) norm(J^(G) (q) v - attach(V, tl: W, tr: G)_d)_2^2 = (J^(G))^+ attach(V, tl: W, tr: G)_d$.

  Minimum-norm when solutions are non-unique; least-squares when overconstrained.

  _Normal least squares (minimize $||A x - b||_2^2$)_

  - If $A$ has full column rank: solve $A^T A x = A^T b$ (e.g., Cholesky) ⇒ $x = (A^T A)^(-1) A^T b$.
  - Equivalently, $x = A^+ b$ with $A^+ := (A^T A)^(-1) A^T$.

  == #fa-icon("sliders") Velocity, Position, Acceleration Constraints (QP)

  With step $h$ and measured $(q, v)$, solve at each control step: $min_(v_n) norm(J^(G) (q) v_n - attach(V, tl: W, tr: G)_d)_2^2$ s.t. $v_"min" <= v_n <= v_"max", q_"min" <= q + h v_n <= q_"max", dot(v)_"min" <= (v_n - v)/h <= dot(v)_"max"$.

  Convex QP improves robustness near limits and singularities.

  == #fa-icon("bullseye") Joint Centering (Nullspace)

  Add a secondary objective projecting into nullspace $P(q)$ of $J(q)$:
  $
    min_(v_n) norm(J^(G) (q) v_n - attach(V, tl: W, tr: G)_d)_2^2
    + epsilon norm(P(q) (v_n - K (q_0 - q)))_2^2, quad epsilon << 1
  $
  Yields a unique solution and draws joints toward nominal $q_0$ without disturbing primary task when unconstrained.

  == #fa-icon("bullseye") Tracking a Desired Pose via Velocity

  Convert pose target $attach(X, tl: A, tr: G)_d$ to twist: $attach(v, tl: A, tr: G)_(A) = ( attach(p, tl: A, tr: G)_d(A) - attach(p, tl: A, tr: G)_(A) ) / h$, $attach(omega, tl: A, tr: G)_(A) = (1 / h) "axisangle"(attach(R, tl: G, tr: G)_d)$. Feed as $attach(V, tl: A, tr: G)_d$ to the QP.

  == #fa-icon("shuffle") Integrability & Alternatives

  Pseudo-inverse paths can be non-integrable (closed task-space loop may end at different $q$). An alternative imposes directional consistency: $max_(v_n, alpha) alpha$ s.t. $J^(G) (q) v_n = alpha attach(V, tl: W, tr: G)_d$, $0 <= alpha <= 1$. Or component-wise scaling in RPY+XYZ via a coordinate map $E(q)$ for teleop.

  _Keyframes, interpolation, differentiation_

  == #fa-icon("key") Keyframes

  Choose gripper keyframe poses and times: initial, pregrasp, grasp, postgrasp, clearance, preplace, place, postplace.

  == #fa-icon("shuffle") Interpolation

  *Positions:* First-order hold (piecewise linear). *Orientations:* SLERP with unit quaternions: $"slerp"(q_0, q_1; t) := (sin((1 - t) theta)/sin(theta)) q_0 + (sin(t theta)/sin(theta)) q_1$, where $theta := arccos(iprod(q_0, q_1))$.

  == #fa-icon("bolt") Differentiate to Velocity

  `PiecewisePose` differentiates to a twist trajectory $attach(V, tl: W, tr: G) (t)$ used by the differential IK controller.

  _Putting it together_

  == #fa-icon("arrows-to-circle") Grasp/Pregrasp via Transforms

  With known $attach(X, tl: W, tr: O)$, choose relative gripper frames $attach(X, tl: O, tr: G_"grasp")$, $attach(X, tl: O, tr: G_"pregrasp")$ with straight approach/retract.
  Example offsets (m): $attach(p, tl: G_"grasp", tr: O) := [0, 0.11, 0]^T$, $attach(p, tl: G_"pregrasp", tr: O) := [0, 0.2, 0]^T$; orientation via $"MakeXRotation"(pi/2) "MakeZRotation"(pi/2)$.

  == #fa-icon("gears") Control

  System diagram: keyframe trajectory -> pose-to-velocity -> differential IK QP (limits + nullspace) -> joint velocity -> integrate to joint position for the low-level position controller.
]

#gatbox(
  title: [#fa-icon("camera") Ch. 04: Geometric Pose Estimation],
)[
  == #fa-icon("up-right-from-square") Pinhole Camera Model
  _Projection/back-projection and RGB-D specifics_
  Given pixel coords $(u, v)$,

  *Projection:* For $vb(P)^C :in RR^3 :quad [u, v, 1]^T = (1/z) K [x, y, z]^T$, camera intrinsics $K := mat(f_x, 0, c_x; 0, f_y, c_y; 0, 0, 1)$.

  *Back-Projection:* Given $(u,v)$ and depth $d$: $vb(P)^C := d K^(-1) [u, v, 1]^T$.

  == #fa-icon("layer-group") Representations
  _Data structures & frame annotation_

  - Depth image $D(u,v)$; Point cloud $cal(P) := Set(vb(P)^C)$; Scene points with frame labels

  == #fa-icon("bullseye") Rigid Alignment on $"SE"(3)$
  _Point-set registration in 3D_
  *Problem:* Given pairs $(vb(p)_i, vb(q)_i)$ in $RR^3$,
  $
    min_(R :in "SO"(3), vb(t) :in RR^3) sum_(i=1)^N norm(R vb(p)_i + vb(t) - vb(q)_i)^2
  $
  *Solution (Kabsch/Umeyama):* center points $overline(vb(p)), overline(vb(q))$; form $tilde(vb(p))_i, tilde(vb(q))_i$; build $W := sum_(i=1)^N tilde(vb(q))_i tilde(vb(p))_i^T$; SVD $W = U Sigma V^T$; set $R^* := U "diag"(1, 1, det(U V^T)) V^T$, $vb(t)^* := overline(vb(q)) - R^* overline(vb(p))$. Weighted: use $W := sum_i w_i tilde(vb(q))_i tilde(vb(p))_i^T$ and weighted means.

  == #fa-icon("arrows-turn-to-dots") Point-to-Plane (small-angle approx)
  _Linearized rotation update for faster convergence_

  Given target normals $vb(n)_i$ at $vb(q)_i$: $min_(R, vb(t)) sum_i ( vb(n)_i^T (R vb(p)_i + vb(t) - vb(q)_i) )^2$.
  Linearize $R approx I + "skew"(vb(theta))$ and solve normal equations for $(vb(theta), vb(t))$.

  == #fa-icon("list-ol") Algorithm (Point-to-Point)
  _Alternate correspondences and pose until convergence_

  1) Initialize $R_0, vb(t)_0$ (e.g., from rough pose or RANSAC)
  2) Correspondences: for each $vb(p)_i$ find nearest $vb(q)_(c(i))$ (kd-tree)
  3) Reject outliers: max distance, normal angle, robust weights
  4) Pose update: solve closed-form registration for $(R_(k+1), vb(t)_(k+1))$
  5) Stop when $norm(vb(t)_(k+1) - vb(t)_k)$ and rotation change are small or max iters reached

  == #fa-icon("down-left-and-up-right-to-center") Point-to-Plane ICP

  Faster convergence near solution; minimize $sum (vb(n)_i^T (R vb(p)_i + vb(t) - vb(q)_i))^2$ via linear least-squares each iteration.

  == #fa-icon("filter") Practical Tips

  - Multi-scale: voxel downsample then refine at higher resolutions
  - Re-estimate normals after filtering; orient via view vector
  - Use max correspondence distance schedule (coarse -> fine)

  == #fa-icon("ban") Outlier Handling
  _Robustness to partial views and clutter_

  - Max correspondence distance; normal compatibility threshold
  - Robust costs (Huber/Tukey) or trimming (keep best rho% pairs)
  - RANSAC on minimal sets (3 non-collinear pairs) to seed pose

  == #fa-icon("object-group") Segmentation
  _Preprocess clouds for registration_

  - Remove dominant plane (table) via RANSAC plane fit
  - Euclidean clustering in 3D, optionally color cues
  - Mask known static geometry from the scene model

  == #fa-icon("table-cells") Soft Correspondence Matrix
  _Soft assignments and EM-style registration_

  - $C :in RR^(N times M)$, rows sum to 1; $C_(i j)$ = belief $vb(p)_i$ matches $vb(q)_j$
  - Alternate: (E) update $C$ from distances; (M) update $(R,vb(t))$ by weighted registration
  - Use Gaussian kernels with bandwidth annealing; add uniform outlier row

  == #fa-icon("mountain") Global Optimization & Non-Penetration
  _Pose from depth with physical constraints_
  *Formulation:*
  $
    min_(X :in "SE"(3))
    sum_i w_i norm(X vb(p)_i - vb(q)_(c(i)))^2
    + lambda sum_("mesh facets") phi_+( d_"signed" (X, "scene") )
  $
  where $phi_+$ penalizes penetration only. Use signed distance fields or mesh distance queries; optionally visibility constraints.

  *Practice:*
  - Initialize from ICP/RANSAC; refine with smooth SDF penalties
  - Jointly estimate multiple objects with mutual non-penetration
  - Leverage solver trust-regions; enforce $"SO"(3)$ via retraction

  == #fa-icon("vector-square") Surface Normal Estimation
  _Estimate surface orientation for point-to-plane and filtering_

  - Cross-product on local grid: finite differences on back-projected rays
  - PCA on $k$-NN: smallest-eigenvector of covariance gives normal
  - Orient consistently: flip to face sensor, then smooth

  // _Practical hooks from course stack_
  // - `RgbdSensor` for synthetic depth; convert to point clouds
  // - Use monogram notation and `attach` for frame annotations
  // - ICP/registration available in examples; collision queries via geometry engine
]

#gatbox(
  title: [#fa-icon("hand") Ch. 05: Grasping & Task Planning],
)[
  _Contact models and quasistatic force balance_

  == #fa-icon("thumbs-up") Contact Models

  *Point Contact w/o Friction (PC)*: $f_n >= 0$, $vb(f)_t = vb(0)$.
  *Point Contact w/ Friction (PCWF)*: Coulomb cone $norm(vb(f)_t) <= mu f_n$.
  *Soft-Finger (SF)*: Adds torsional moment $abs(tau_n) <= mu_r f_n$ via limit surface.

  == #fa-icon("scale-balanced") Quasistatic Equilibrium

  Let $m$ contacts, stacked contact force vector $vb(f)_c :in RR^(k m)$ (with $k in {1,2,3}$ per model). The grasp matrix $G :in RR^(6 x k m)$ maps to object wrench: $vb(w) := G vb(f)_c = [vb(f); vb(tau)]$
  Force balance with external wrench $vb(w)_"ext"$:
  $G vb(f)_c + vb(w)_"ext" = vb(0)$

  Feasible if each contact satisfies its friction/torque limits, e.g. PCWF: $f_n >= 0$, $norm(vb(f)_t) <= mu f_n$ (often linearized by a pyramid).

  _Coverage in wrench space and quantitative quality_

  == #fa-icon("cone") Contact/Grasp Wrench Cones

  Each contact i generates a convex cone of wrenches $cal(C)_i := Set(G_i vb(f)_i : (vb(f)_i, "constraints"))$.
  The Grasp Wrench Set (GWS) is the Minkowski sum $cal(W) := cal(C)_1 + dots + cal(C)_m$ (convex for linearized cones).

  == #fa-icon("lock") Force Closure vs Form Closure

  *Force closure:* $cal(W)$ contains a neighborhood of the origin $vb(0)$ $=>$ can resist any small wrench. Equivalent: the convex cone $"cone"(cal(W))$ spans $RR^6$.
  *Form closure:* Purely geometric immobilization (typically more contacts, e.g. frictionless $>=$ 7 in 3D). In practice we seek force closure.

  == #fa-icon("ruler") Quality Metrics (epsilon-metric)

  With unit-bounded contact efforts, define the largest ball around $vb(0)$ contained in $cal(W)$: $q_epsilon := max_(scal(e) >= 0) min_(norm(vb(w)) = 1) vb(w)^T ( sum_i G_i vb(f)_i )$.

  Intuition: worst-case unit wrench margin resisted by the grasp. Larger is better; sensitive to contact locations, normals, and friction.

  _Pairs of contacts aligned with friction cones_

  == #fa-icon("angles-up") Condition

  Two surface points with outward normals $vb(n)_1, vb(n)_2$ and line-of-centers direction $vb(d)$ are antipodal if $vb(d)$ lies within both friction cones:
  $
    angle(-vb(n)_1, vb(d)) <= arctan(mu) quad "and" quad angle(vb(n)_2, vb(d)) <= arctan(mu)
  $
  For frictionless: require $vb(d)$ colinear with $-vb(n)_1$ and $vb(n)_2$.

  == #fa-icon("check") Use

  Simple, effective heuristic for parallel-jaw grasps; robust under moderate pose error when cones are wide (larger $mu$).

  _From geometry to feasible, high-quality grasps_

  == #fa-icon("mountain-sun") Geometry Cues

  From mesh/point cloud: estimate normals and curvature; sample handles/edges; sample antipodal pairs on locally parallel patches; for suction, prefer smooth, planar regions with sufficient area and reachable normal.

  == #fa-icon("filter") Feasibility Filters

  - Collision-free closing region; gripper width limits; approach clearance.
  - Reachability/IK with margin; avoid joint limits/singularities; plan approach and retreat.

  == #fa-icon("ranking-star") Ranking

  Score via $q_epsilon$, normal alignment, distance to edges, friction margin $arctan(mu) - angle$; penalize occlusion/uncertainty. Choose top-K diverse grasps, then refine with local optimization.

  _Composing grasping with perception, motion, and recovery_

  == #fa-icon("shuffle") Finite-State Machines (FSM)

  Discrete modes (Detect → Plan → Grasp → Place) with guarded transitions. Simple and explicit, but can become brittle and hard to scale.

  == #fa-icon("tree") Behavior Trees (BT)

  Nodes tick top-down. Core controls: Sequence (fails fast on first child failure), Fallback/Selector (succeeds on first child success), Parallel, Decorators (timeouts, retries), and leaf Actions/Conditions. Advantages: modularity, reactivity, reuse. Blackboard shares state.

  == #fa-icon("recycle") Robustness Patterns

  Timeouts with retry/fallback; perception refresh on failures; re-plan on IK/collision failure; grasp re-ranking; escalate to place-in-bin if precise place fails.

  _Polyhedral cones enable LP-based checks_

  == #fa-icon("code-compare") Polyhedral Approximation

  For each contact i, approximate the circular cone by r generators $D_i := [vb(d)_(i 1) .. vb(d)_(i r)]$ (unit directions in tangential space and normal). Contact force $vb(f)_i := D_i scal(alpha)_i$ with $scal(alpha)_i >= vb(0)$ yields linear constraints. Stack $W := [G_1 D_1 .. G_m D_m]$ and $scal(alpha) := [scal(alpha)_1; ..; scal(alpha)_m]$.

  == #fa-icon("square-root-variable") Equilibrium as LP

  Feasibility under external wrench $vb(w)_"ext"$:
  $
    "find" scal(alpha) >= vb(0) quad "s.t." quad W scal(alpha) + vb(w)_"ext" = vb(0)
  $
  If feasible, the grasp can statically resist $vb(w)_"ext"$. To add joint torque limits $abs(vb(tau)_j) <= tau_("max", j)$ use $vb(tau) = J^T F$ where $F$ stacks the contact forces, giving extra linear inequalities.

  _Worst-case wrench margin from a unit-effort GWS_

  == #fa-icon("certificate") Ferrari–Canny epsilon-Metric (Computation)

  == #fa-icon("bullseye") Unit GWS

  Using polyhedral cones with $sum scal(alpha) <= 1$ defines a compact $cal(W)_1 := Set(W scal(alpha) : scal(alpha) >= vb(0), sum scal(alpha) <= 1)$.

  == #fa-icon("ruler-vertical") epsilon via Directional LP

  Sample unit directions $vb(u)_k$ on $S^5$ and solve
  $
    "minimize" vb(u)_k^T vb(w) quad "s.t." vb(w) in cal(W)_1
  $
  Then $q_epsilon = min_k ( - "optval"_k )$. More exactly, with an H-representation $A vb(w) <= vb(b)$ of $cal(W)_1$, $q_epsilon = min_i ( b_i / norm(vb(a)_i) )$ where $vb(a)_i^T vb(w) <= b_i$ are facets.

  _Mapping contact forces to joint torques and constraints_

  == #fa-icon("diagram-project") Wrench–Torque Relations

  Stack contact forces $F$; object wrench $vb(w) = G F$; hand joint torques $vb(tau) = J^T F$. Add bounds $abs(vb(tau)) <= vb(tau)_"max"$, fingertip force bounds, and closure region collisions to the LP/QP for realistic feasibility.

  _Normal capacity from vacuum; shear from friction/seal_

  == #fa-icon("circle-dot") Capacities

  Normal: $F_n^("max") := (Delta P) A$ from pressure differential $Delta P$ and cup area $A$. Shear: $F_t^("max") := mu_s F_n^("max")$ (surface friction). Peel/tilt moment limit roughly $M_n^("max") ~ k A Delta P r$ (cup stiffness k, effective radius r). Require approach aligned with surface normal and sealable patch.

  _Efficient sampling and testing on depth data_

  == #fa-icon("list-check") Procedure

  1) Estimate normals via PCA in local neighborhoods; smooth outliers.
  2) Sample candidates on near-parallel patches; enforce gripper width and thickness.
  3) For pair (p1, p2) with direction $vb(d)$, test $angle(-vb(n)_1, vb(d)) <= arctan(mu)$ and $angle(vb(n)_2, vb(d)) <= arctan(mu)$.
  4) Check collision of closing region and approach clearance.
  5) Score with friction margin and distance-to-edges; deduplicate by pose.

  _Tick outcomes and control flow_

  == #fa-icon("tree") Execution & Robustness

  - *Node statuses:* success/failure/running; Sequence returns first non-success; Selector returns first success; Decorators modify child status; Blackboard shares state (pose, IK, grasp-index).
  - *Robust pattern:* Detect → Sample/Rank grasps → Try K: PlanIK → Execute; on failure: try next; if all fail: refresh perception/retry; fallback to place-in-bin.

  _Robust model fitting via random sampling_

  == #fa-icon("bolt") Model and Inliers

  Plane: $vb(n)^T vb(x) + d = 0$ with $norm(vb(n)) = 1$. Inlier test for point $vb(x)_i$:
  $
    abs(vb(n)^T vb(x)_i + d) <= tau
  $
  Minimal sample size: $s := 3$ (non-collinear points define a plane).

  == #fa-icon("gauge") Iteration Budget

  Given desired success prob $p in [0, 1]$ and inlier ratio $w in [0, 1]$:
  $
    N_"iters" := ceil(ln(1 - p) / ln(1 - w^s))
  $
  Refit with all inliers, select model with largest consensus set.

  _Handling mismatches and robustness_

  == #fa-icon("vector-square") Correspondences and Objective

  Standard ICP minimizes
  $
    sum_i norm(O vb(p)^m_(j(i)) - W vb(p)^s_i)^2
  $
  where $j(i)$ is nearest-neighbor. With outliers, use trimming (keep a fraction of smallest residuals) or robust losses (Huber/Tukey) via weighted least squares.

  == #fa-icon("triangle-exclamation") Practical Notes

  - Initialize with reasonable pose; re-estimate correspondences after each update.
  - Reject pairs beyond a max distance; enforce normal consistency when available.

  _For a block of mass m on slope angle theta_

  == #fa-icon("scale-balanced") Forces

  $
    f_t = m g sin(theta), quad f_n = m g cos(theta)
  $
  No-slip condition: $mu >= tan(theta)$.

  _PCA in local neighborhoods_

  == #fa-icon("square-root-variable") Covariance and Normal

  For neighborhood points ${ vb(p)_i }$, mean $vb(p)_bar := (1/k) sum_i vb(p)_i$ and covariance
  $
    W := sum_i (vb(p)_i - vb(p)_bar) (vb(p)_i - vb(p)_bar)^T
  $
  The eigenvector of $W$ with the smallest eigenvalue is the surface normal. Orient consistently (e.g., toward the sensor) if needed.

  _Tangency conditions and Hessian interpretation_

  == #fa-icon("angles-up") Conditions

  Let boundary be $vb(p)(t)$ with tangent $vb(tan)(t) := vb(p)'(t)$. A frictionless antipodal pair $(t_1, t_2)$ satisfies
  $
    vb(tan)(t_1)^T ( vb(p)(t_2) - vb(p)(t_1) ) = 0, quad
    vb(tan)(t_2)^T ( vb(p)(t_1) - vb(p)(t_2) ) = 0
  $
  These imply the gradient of a suitable alignment energy in $(t_1, t_2)$ is zero (the pset's $partial E / partial t = [0, 0]$ observation).

  == #fa-icon("ruler") Hessian and Preference

  - Local minima → concave points, local maxima → convex points, saddles → mixed curvature.
  - Preferred grasps are at convex (local maxima) regions to avoid penetration.
]

#colbreak()
#gatbox(
  title: [#fa-icon("screwdriver-wrench") Ch. 06: Motion Planning],
)[
  _Rich costs and constraints over joint configurations_

  == #fa-icon("diagram-project") Problem View

  $"Given" f_"kin" : q mapsto X_G, quad "solve for" q "with costs/constraints"$

  _Formulations_

  $"Nominal IK:"$ $min_(q) norm(q - q_0)^2$ s.t. $attach(X, tl: G, tr: O) = f_"kin"(q)$ (pose match), $q in [q_"min", q_"max"]$ (joint limits), $d_"min" (q) >= 0$ (non-penetration).

  $"Differential IK:"$ one SQP/least-squares step in $Delta q$ around $q_t$; good locally, cannot switch homotopy classes.

  _Constraint Templates_

  $"Position:" p_G (q) = p^* quad "Orientation:" R_G (q) = R^* quad "Distance:" d_"min" (q) >= 0 quad "Joint:" q in [q_"min", q_"max"]$

  _Differential IK (Least-Squares)_

  $min_(Delta q) norm(J(q_t) Delta q - v^*)^2 + lambda norm(Delta q)^2$ s.t. $A Delta q <= b$ (vel/avoidance bounds). $=> Delta q = (J^T J + lambda I)^(-1) J^T v^*$ (damped pseudo-inverse).

  _Guidance_

  - Keep objectives simple (joint-centering); encode hard requirements as constraints.
  - Write minimal constraints (e.g., partial orientation, point-on-line contacts).
  - Collision constraints are nonconvex; solvers like SNOPT often work at interactive rates but offer no guarantees.

  // === Global IK (offline)

  // Mixed-integer convex or SDP relaxations can deliver robustness/globality for restricted cost/constraint sets; slower but valuable for dataset generation and workspace analysis.

  _Optimize a continuous joint-space trajectory with time-scaling_

  == #fa-icon("clock") Trajectory Optimization

  $min_(alpha, T)\, T$ s.t. $attach(X, tl: G_"start", tr: O) = f_"kin"(q_alpha (0))$, $attach(X, tl: G_"goal", tr: O) = f_"kin"(q_alpha (T))$, $abs(dot(q_alpha) (t)) <= v_"max"$ for all $t :in [0, T]$.

  _B-spline Path + Time Rescaling_

  - Path $r(s)$, $s in [0,1]$ as B-spline; trajectory: $q(t) := r(t/T)$.
  - Bases: $r(s) = sum_i N_(i,k) (s) c_i$ with control points $c_i$.
  - Derivatives: $d/(d s) r(s) = sum_i N_(i,k-1) (s) D_i$ where $D_i := k/(u_(i+k) - u_i) (c_i - c_(i-1))$.
  - Convex hull: $r(s)$ lies in convex hull of active $c_i$ $=>$ box constraints on $c_i$ imply forall $s$ bounds on $r(s)$.

  _Velocity Constraints via Time-Scaling_

  $s := t/T => dot(q) (t) = dv(r, s) (s) (1/T)$. Enforce $abs(dot(q) (t)) <= v_"max"$ for all $t$ by linear bounds on $D_i$: $abs(D_i) <= v_"max" T$ for all $i$. Higher derivatives scale as $T^(-2)$, $T^(-3)$, ... and become nonlinear in $(c_i, T)$; they can be constrained at samples or relaxed.

  // == #fa-icon("gears") Practicalities

  // - Use collision constraints at sampled $s_i$; verify densely post-opt.
  // - Local minima are common; seed well (e.g., no-collision solve → warm start with collisions).
  // - Related methods: CHOMP, STOMP, KOMO (augmented Lagrangian).

  _Global exploration with probabilistic completeness_

  == #fa-icon("tree") Sampling-based Planning

  $"RRT (basic):"$ sample $q_"rand"$, find $q_"near" = arg min_q d(q, q_"rand")$, step toward by $eta$, keep if collision-free. Good in high-dim; jerky and suboptimal; suffers in bug traps.
  $"Key knobs:"$ step size $eta$, goal bias $p_"goal"$, metric $d(., .)$, local planner.
  $"RRT*:"$ rewiring for asymptotic optimality using radius $r_n ~ (log n / n)^(1/d)$.
  $"RRT-Connect:"$ bidirectional; grow trees from start and goal; connect repeatedly extends toward the other tree by step $eta$ until blocked; very fast first solution; not asymptotically optimal.

  $"AO-RRT-Connect (AORRTC):"$ bidirectional RRT\* with rewiring in balls of radius $r_n ~ (log n / n)^(1/d)$; attempts connections while rewiring both trees; asymptotically optimal; often keeps RRT-Connect's fast first path.

  _PRM_

  Offline roadmap: sample collision-free milestones; connect $k$-NN or within radius $r$ when straight segment is collision-free; online query via shortest path. Needs post-smoothing for curvature/dynamics.

  _Post-processing_

  Shortcutting and anytime B-spline smoothing; tune distance metrics and collision checking for speed.

  _Bridge global graph search with continuous convex optimization_

  == #fa-icon("layer-group") GCS over Convex Sets

  Replace PRM points/edges with convex regions and continuous decisions at vertices/edges. Solve shortest path over a graph of convex sets via a strong convex relaxation (often tight with rounding).

  _Transcription for Kinematic Planning_

  - Assume convex decomposition of collision-free C-space (justified below).
  - At each visited region, choose two points so line/curve lies within region; enforce equality at overlaps to stitch segments.
  - Use Bézier curves per region + time-scaling; impose convex constraints for continuity and velocity limits that hold forall $t$.

  _Variables and Constraints_

  $"Per vertex (region)" V: "pick" (q_"in", q_"out") in Set(V) times Set(V). "Per edge" (U,V): "enforce" q_"out"^U = q_"in"^V$ when traversed; add convex arc-length/time costs and derivative bounds via Bézier control points.

  _Objectives and Constraints_

  - Costs: duration $T$, path length upper bound, energy $"int" norm(dot q(t))_2^2 d t$.
  - Constraints: derivative continuity, strict velocity bounds forall $t$, initial/final states, additional convex bounds.

  // === Workflow

  // Solve convex relaxation, then round and optionally refine with nonlinear constraints only on the selected path for high fidelity.

  _Inflate samples into certified convex C-space regions_

  == #fa-icon("shapes") IRIS & Region Construction

  - IRIS (Euclidean or C-space): alternate separating hyperplanes and MVEE to inflate regions.
  - IRIS-NP/IRIS-NP2: nonlinear or improved pipelines; fast, probabilistic.
  - IRIS-ZO: zeroth-order, trivially parallel.
  - SOS/Algebraic kinematics: rigorous certificates via polynomials; slower but sound (uses stereographic projection coordinates).

  _Construction Tips_

  Use minimum clique cover over visibility graph for efficient covering; seed with IK solutions, teleop demos, or other planner rollouts to cover "important" volumes in high DOF.

  _Iteration Sketch (IRIS)_

  1) Fix obstacle set; solve for separating hyperplanes that keep a convex polytope $P$ collision-free.
  2) Fix $P$; solve MVEE to maximize enclosed ellipsoid volume.
  3) Inflate/clip and repeat until convergence.

  _Trade-offs_

  - IK/Diff-IK: fast; local vs global; use constraints not penalties.
  - Kinematic TrajOpt: rich constraints, online-capable; local minima; sample collisions sparsely + dense verification.
  - RRT/PRM: global completeness notion; limited curvature/dynamics; requires smoothing.
  - GCS: global structure + continuous constraints; strong relaxations; needs convex decomposition (IRIS family).
]

#gatbox(
  title: [#fa-icon("people-carry-box") Ch. 07: Mobile Manipulation],
)[
  _Extends table-top manipulation with mobility; raises new perception, planning, and simulation challenges._

  == #fa-icon("bullseye") Scope & Motivation

  *Ambition boost:* Mobility enables in-home tasks across rooms and scenes. Many tools carry over; critical differences arise in perception, state estimation, and navigation.

  _Partial views, unknown environments, and robot state estimation become central._

  == #fa-icon("eye") Perception, Mapping & State Estimation

  *One-sided observations:* Head-mounted sensors often see only one side of objects; antipodal grasp heuristics need completion.
  *Learning is fundamental:* Infer occluded geometry via data (shape completion, semantics). Move sensors to reduce uncertainty (active perception; planning under uncertainty).

  _Mathematical framing (shape completion & VOI)_

  $
    O := "object shape"; Z := "current view"
  $

  $
    hat(O) := arg max_O Pr[O | Z, cal(D)]
    = arg max_O Pr[Z | O] Pr[O | cal(D)]
  $

  $
    a^* := arg max_a EE[ U("bel"') - U("bel") | a ]\
    "bel"' := "posterior after active view " a
  $

  _Unknown/Dynamic Environments_

  *Representation needs:* Fast collision queries, distance fields, scalable updates from raw depth/RGB-D.
  *Voxel grids:* Discretize space; maintain occupied/free/probabilistic occupancy. Efficient sphere–voxel queries; easy parallelization.
  *OctoMap:* Octree-based multi-resolution mapping; probabilistic updates incl. free space. Good for large scenes.

  _Occupancy updates (log-odds)_

  $
    L_t(v) := ln(Pr["occ"_v | z_(1:t)] / (1 - Pr["occ"_v | z_(1:t)]))\
    L_t(v) = L_(t-1)(v) + l(z_t | v) - l_0
    Pr["occ"_v | z_(1:t)] = 1 / (1 + exp(-L_t(v)))
  $

  where $l(z_t | v)$ is the inverse sensor model contribution for voxel $v$ along ray $z_t$ and $l_0 := ln(p_0 / (1-p_0))$.

  _Sphere–voxel distance query_

  $
    d(q) := min_(v in cal(V)_"occ") "dist"_2( c_s(q), v ) - r_s
  $

  $
    d(q) > 0 => "collision-free; enables fast clearance costs via EDT"
  $

  _Robot State Estimation (Localization)_

  *Beyond fixed-base:* Must estimate $attach(X, tl: W, tr: C)$ and base pose; wheel odometry alone drifts (slip). Fuse IMU, wheels, lidar/RGB-D/RGB.
  *Classics:* Recursive Bayes filters and smoothing (e.g., pose-graph SLAM / iSAM).
  *Trends:* Strong visual(-inertial) odometry; monocular depth; dense 3D reconstruction (e.g., NeRF, Gaussian splatting).

  _Bayes filter (discrete-time)_

  $
    "bel"_t (x) := eta Pr[z_t | x] sum_(x') Pr[x | u_t, x'] "bel"_(t-1)(x')
  $

  _EKF (nonlinear Gaussian models)_

  $
    x_(t+1) = f(x_t, u_t) + w_t, quad
    z_t = h(x_t) + v_t, \
    w_t ~ cal(N) (0, Q), quad
    v_t ~ cal(N) (0, R)
  $

  $
    A_t := pdv(f, x) |_(x_t), quad
    W_t := pdv(f, w) |_(x_t), quad
    H_t := pdv(h, x) |_(x_t)
  $

  $
    P_(t+1|t) = A_t P_t A_t^top + W_t Q W_t^top
  $

  $
    K_(t+1) = P_(t+1|t) H_(t+1)^top (
      H_(t+1) P_(t+1|t) H_(t+1)^top + R
    )^(-1)
  $

  $
    x_(t+1) = x_(t+1|t) + K_(t+1) (
      z_(t+1) - h(x_(t+1|t))
    )
  $

  $
    P_(t+1) = (I - K_(t+1) H_(t+1)) P_(t+1|t)
  $

  _Pose-graph SLAM objective_

  $
    X := (x_0, ..., x_T),
    J(X) := sum_i norm(r_i (X))_(Sigma_i^(-1))^2,
    r_i := "between-factor error"
  $

  _Same kinematic tools, with added base DOFs and occasionally nonholonomic constraints._

  == #fa-icon("robot") Base & Kinematics

  *iiwa + {x,y,z} prisms + yaw:* Treat base as extra joints; IK, trajopt, RRT/PRM apply directly. Local trajopt scales well with dimension; sampling planners need care as DOFs grow.
  *Continuous joints:* Handle wrap-around in metrics/extend; in GCS/IRIS use local coordinates with <= $pi$ domain for wrapping joints to avoid long-way paths.

  _Mobile-base IK (holonomic base)_

  Variables: $q := (x_b, y_b, z_b, "yaw"_b, q_"arm")$

  $
    min_q w_p norm(p_"ee" (q) - p_"goal")_2^2 + w_R norm(log(R_"goal"^top R_"ee" (q)))_2^2 + w_c norm(q - q_0)_2^2 \
    s.t. q_"min" <= q <= q_"max", d_"min" (q) >= d_"safe"
  $

  Orientation error uses rotation log-map. $d_"min"$ is signed clearance (e.g., EDT over voxels).

  _Angle wrap metric_

  $
    d_theta(a, b) := "atan2"(sin(a - b), cos(a - b))
  $

  _Wheeled Bases_

  *Plan with no-slip; track with feedback.*
  - Holonomic drives: e.g., mecanum/omni; direct $(x, y, "yaw")$ commands feasible.
  - Nonholonomic drives: differential drive, Dubins (forward-only turns), Reeds–Shepp (with reverse). Distance/extend must respect constraints.

  _Differential-drive kinematics and Pfaffian constraint_

  $
    dot(x) = v cos(theta), dot(y) = v sin(theta), dot(theta) = omega \
    A(q) dot(q) = 0, A(q) := [ -sin(theta), cos(theta), 0 ] => -sin(theta) dot(x) + cos(theta) dot(y) = 0
  $

  _Dubins / Reeds–Shepp primitives_

  $
    P := "sequences of " L/R/S " minimizing path length subject to turn-rate bounds" \
    kappa_"max" := v / R_"min", "curvature bound"
  $

  Distance/extend in planners must compose feasible primitives and respect $kappa_"max"$.

  _Legged Bases_

  *As a user of platform APIs (e.g., Spot):* Often exposed as holonomic base; low-level balance handled by platform. Watch COM/balance-induced deviations from commanded path; rough terrain introduces richer constraints.

  _Beyond engines; need environments and assets for mobile tasks._

  // == #fa-icon("gears") Engines & Content

  // *Engines:* Drake, MuJoCo, Isaac/Omniverse (rendering + physics).
  // *Aggregators:* Sapien, Behavior-1K, Habitat 3.0 provide scenes/assets/tasks; aim to interoperate and bring problems into Drake stacks for perception/planning/control.

  // _Range sensor simulation (essentials)_

  // $
  //   z = "cast"("Ray"(W, "pose"_s), "Scene")\
  //   z' = z + epsilon, epsilon ~ cal(N) (0, sigma^2), "per-ray noise model"
  // $

  // _Problems unique/salient for mobile platforms._

  // _Mapping in Addition to Localization_

  // *Build maps while localizing:* Maintain/update occupancy structures (voxel/octree) to enable fast planning queries.

  // _Online occupancy mapping_

  // $
  //   cal(M)_t := "map state (log-odds)"
  //   cal(M)_t = "Update"(cal(M)_(t-1), z_t, x_t),\ "ray-casting with inverse sensor model; preserves sparsity in octrees"
  // $

  // _Traversable Terrain Identification_

  // *Classify free vs risky terrain:* From depth/semantics; feed into planners/costmaps and base controllers.

  // _Heuristics and costmaps_

  // $
  //   "slope"(p) := arccos(n(p) dot e_z), "roughness"(p) := var(z(N(p))) \
  //   "traversable"(p) := ( "slope"(p) <= alpha_"max" ) and ( "roughness"(p) <= r_"max" ) \
  //   J_"path" := "int"_(s in "path") [ w_d + w_inf / ( d("path"(s)) + epsilon ) ] dd(s), "distance-transform inflation"
  // $

  _Exercises (What to Practice)_

  == #fa-icon("boxes-stacked") Simulation Set-Up

  *Analyze collision geometry of SDFs; compose a manipulation scene.*

  == #fa-icon("arrows-left-right-to-line") Mobile-Base IK

  *Solve IK with and without fixed base; removing base position constraints simplifies optimization.*
]

#gatbox(
  title: [#fa-icon("toolbox") Ch. 08: Manipulator Control],
)[
  _Torque-level controllers that execute higher-level motion/force commands_

  == #fa-icon("sliders") Core Systems

  *PD (PidController):* $u <- K_p (q_d - q) + K_d (dot(q)_(d) - dot(q)) + K_i integral (q_d - q)$. In manipulation, typically set $K_i := 0$ (avoid windup).

  *Joint Stiffness:* $u <- -tau_g (q) + K_p (q_d - q) + K_d (dot(q)_(d) - dot(q))$ (gravity comp + PD). Removes steady-state error under constant loads.

  *Inverse Dynamics Control:* $u <- M(q) ddot(q)_(d) + C(q, dot(q)) dot(q) - tau_g (q)$. Choose $ddot(q)_(d) := ddot(q)_("ref") + K_p (q_("ref") - q) + K_d (dot(q)_("ref") - dot(q))$ for tracking.

  *Spatial Force Control:* Command desired Cartesian force at contact/end-effector; maps to joint torques.

  *Spatial Stiffness (Operational Space):* Program end-effector to behave like mass-spring-damper in task frame; handle nullspace with joint objectives.

  _Start with 2D point-finger of mass $m$; gravity along $-z$; contact force $f^(F_c)$_

  == #fa-icon("gauge-high") PD vs Gravity Comp vs Inverse Dynamics

  *PD:* $m ddot(z) = -m g + k_p (z_d - z) + k_d (dot(z)_(d) - dot(z))$ => steady-state error under constant $m g$:
  $
    tilde(z) := z_d - z = m g / k_p.
  $

  *Gravity-comp Stiffness:* $u <- -tau_g + K_p (q_d - q) + K_d (dot(q)_(d) - dot(q)) => m ddot(z) = k_p tilde(z) + k_d dot(tilde(z))$ (no steady-state error to gravity).

  *Inverse Dynamics (with accel feedforward):*
  $
    u <- -tau_g + m[ ddot(q)_(d) + K_p (q_d - q) + K_d (dot(q)_(d) - dot(q)) ] \
    => ddot(tilde(z)) + k_d dot(tilde(z)) + k_p tilde(z) = 0 quad "mass-spring-damper on error".
  $

  _Direct Force Control_

  == #fa-icon("hand") Quasi-static idea

  With small accelerations in contact: $f^(F_c) = -m g - u$ => choose
  $
    u <- -m g - f^(F_c)_("desired").
  $
  Off-contact, same command accelerates the finger toward contact (useful for autonomous approach without precise geometry).

  == #fa-icon("balance-scale") Free-space vs Contact: Steady State

  Controller (1D): $u := k_p (x_d - x) - f^(F_c)_("desired")$.
  - Free space ($f^(F_c) = 0$), steady state ($dot(x) = ddot(x) = 0$):
  $
    0 = k_p (x_d - x) - f^(F_c)_("desired") quad => quad x - x_d = - f^(F_c)_("desired") / k_p.
  $
  Cannot have $x = x_d$ and nonzero $f^(F_c)_("desired")$ simultaneously.
  - In contact: wall reaction $f^(F_c)$ balances $f^(F_c)_("desired")$ ⇒ zero steady-state error achievable.
]

#gatbox(
  title: [#fa-icon("magnet") Ch. 08 (cont.): Impedance, Hybrid, and OSC],
)[
  _Indirect Force Control (Stiffness/Impedance)_

  == #fa-icon("spring") Target behavior in Cartesian coordinates

  For end-effector $F$ with position $p^F := [x, z]^T$ and velocity $v^F$:
  $
    m dot(v^F) + K_d v^F + K_p (p^F - p^(F_d)) = f^F.
  $
  Implement via gravity-comp PD at joints or operational-space control. Passivity (with $K_d$ "positive definite") grants robust stability under unknown environments.

  _Hybrid Position/Force Control_

  == #fa-icon("ruler") World frame form

  $
    u <- -tau_g + [ k_p (x_d - x) + k_d (dot(x)_(d) - dot(x)) ; -f^F_("desired", W_z) ].
  $

  == #fa-icon("up-down-left-right") Contact-frame form

  With rotation $attach(R, tl: W, tr: C)$:
  $
    u <- -tau_g + attach(R, tl: W, tr: C) [ k_p (p^(F_d)_(C_x) - p^F_(C_x)) + k_d (v^(F_d)_(C_x) - v^F_(C_x)) ; -f^F_("desired", C_z) ].
  $

  _From joint-space dynamics to operational space and nullspace control_

  == #fa-icon("book") Dynamics

  $
    M(q) ddot(q) + C(q, dot(q)) dot(q) = tau_g(q) + u + sum_(i) J_i^T (q) f^(c_i).
  $

  == #fa-icon("wand-magic-sparkles") Inverse Dynamics / Computed Torque

  $
    u <- M ddot(q)_(d) + C dot(q) - tau_g, quad ddot(q)_(d) := ddot(q)_("ref") + K_p (q_("ref") - q) + K_d (dot(q)_("ref") - dot(q)).
  $

  == #fa-icon("tornado") Joint Stiffness (iiwa interface)

  $
    u <- -tau_g(q) + K_p (q_d - q) + K_d (dot(q)_(d) - dot(q)) + tau_("ff").\ "Diagonal" K_p, K_d ; "use" tau_("ff") "for Cartesian forces"
  $

  == #fa-icon("compass") Operational Space (Cartesian Stiffness)

  Let $E$ be end-effector, $p^E = f_("kin")(q)$, $v^E = J(q) dot(q)$. Using $u := J^T attach(f, tl: B, tr: E)_(u)$ and assuming only contact at $E$:
  $
    M_E(q) ddot(p)^E + C_E(q, dot(q)) dot(q) = attach(f, tl: B, tr: E)_(g)(q) + attach(f, tl: B, tr: E)_(u) + attach(f, tl: B, tr: E)_("ext"), \
    M_E := (J M^(-1) J^T)^(-1), quad C_E := M_E (J M^(-1) C - dot(J)), quad attach(f, tl: B, tr: E)_(g) := M_E J M^(-1) tau_g.
  $
  Choose
  $
    attach(f, tl: B, tr: E)_(u) <- -attach(f, tl: B, tr: E)_(g) + K_p (p^(E_d) - p^E) + K_d (dot(p)^(E_d) - dot(p)^E)
    u <- J^T attach(f, tl: B, tr: E)_(u)
  $
  to realize spring-damper behavior at the end-effector.

  == #fa-icon("layer-group") Nullspace Joint Objectives

  With dynamically consistent nullspace projector $P$:
  $
    u <- J^T attach(f, tl: B, tr: E)_(u) + P[ K_(p "joint") (q_0 - q) - K_(d "joint") dot(q) ].
  $
]

#gatbox(
  title: [#fa-icon("screwdriver-wrench") Ch. 08 (cont.): Tuning, Flip-Up, RCC],
)[
  _Tuning and Error Dynamics_

  == #fa-icon("chart-line") Point-mass intuition

  Error ODE: $ddot(tilde(z)) + k_d dot(tilde(z)) + k_p tilde(z) = 0$. Match to $ddot(e) + 2 zeta omega_n dot(e) + omega_n^2 e = 0$:
  $
    k_d := 2 zeta omega_n, quad k_p := omega_n^2.
  $

  == #fa-icon("arrows-left-right") Joint-space effective inertia

  Around configuration $q$, along joint $i$ with inertia $M_(i i)$, the closed-loop approx is $M_(i i) ddot(tilde(q)_i) + K_(d i i) dot(tilde(q)_i) + K_(p i i) tilde(q)_i = 0$ so choose
  $
    K_(d i i) := 2 zeta omega_n M_(i i), quad K_(p i i) := omega_n^2 M_(i i).
  $

  == #fa-icon("crosshairs") Task-space (operational) tuning

  Using $M_E := (J M^(-1) J^T)^(-1)$, set
  $
    K_d := 2 zeta omega_n M_E, quad K_p := omega_n^2 M_E,
  $
  to achieve approximately isotropic second-order error dynamics at the end-effector.

  _Force-Based Flip-Up (Constraints)_

  == #fa-icon("cone") Friction cones

  Finger-on-box at contact frame $C$ and ground at corner frame $A$:
  $
    f^(B_C)_("finger", C_z) >= 0, quad abs(f^(B_C)_("finger", C_x)) <= hat(mu)_(C) f^(B_C)_("finger", C_z),
  $
  $
    f^(B_A)_("ground", A_z) >= 0, quad abs(f^(B_A)_("ground", A_x)) <= hat(mu)_(A) f^(B_A)_("ground", A_z).
  $

  == #fa-icon("rotate") Torque about pivot

  About $A$: $tau^(B_A)_("total", W_y) := tau^(B_A)_("gravity", W_y) + tau^(B_A)_("ground", W_y) + tau^(B_A)_("finger", W_y)$. With $tau_("ground") = 0$ about its own point,
  $
    + tau^(B_A)_("total", W_y) = tau^(B_A)_("gravity", W_y) + tau^(B_A)_("finger", W_y) > 0 quad "to flip up".
  $

  == #fa-icon("equals") Constrained least-squares control

  Given estimate $hat(theta)$ and quasi-static balance at $B$:
  $
    min_( f^(B_C)_("finger", C),\, f^(B_A)_("ground", A) ) abs(tau^(B_A)_("finger", W_y) - "PID"(theta_d, hat(theta)))^2,
  $
  subject to friction-cone inequalities above and force balance
  $
    f^(B_A)_("ground", A) + hat(f)^(B_A)_("gravity", A) + f^(B_C)_("finger", A) = 0.
  $

  == #fa-icon("square-root-variable") Useful bounds (small-angle, square box)

  About $A$ with $C$ at distance $L$ and com at $L/2$:
  $
    f^(B_C)_("finger", C_x) L >= m g (L/2) quad => quad f^(B_C)_("finger", C_x) >= (m g)/2.
  $
  Finger friction: $abs(f^(B_C)_("finger", C_x)) <= mu_(C) f^(B_C)_("finger", C_z)$ ⇒ lower bound on normal:
  $
    f^(B_C)_("finger", C_z) >= (m g) / (2 mu_(C)).
  $
  Ground no-slip at $A$: $abs(f^(B_A)_("ground", A_x)) <= mu_(A) f^(B_A)_("ground", A_z)$ with
  $f^(B_A)_("ground", A_x) = - f^(B_C)_("finger", C_x)$ and $f^(B_A)_("ground", A_z) = m g - f^(B_C)_("finger", C_z)$ gives an upper feasibility condition coupling $mu_(A)$ and $f^(B_C)_("finger", C_z)$.

  _Hybrid Control: Feasibility_

  - Require commanded normal force within friction limits at sticking contacts and below slip threshold at sliding contacts.
  - For planar push with normal along $z$: need $abs(f_("tangent")) <= mu f_("normal")$ at sticking interface; for sliding interface, target $abs(f_("tangent")) approx mu f_("normal")$ with direction opposing slip.
  - Book-drag condition (gripper/table): favor $mu_("gripper") > mu_("table")$ to stick at gripper while sliding on table (ratio threshold near $1$).

  _Reflected Inertia and Gearing_

  == #fa-icon("arrows-rotate") Reflected quantities (Gearboxes)

  $
    J_("reflected") := N^2 J_("load"), quad b_("reflected") := N^2 b_("load"), quad tau_("motor") = (1/N) tau_("load").
  $
  Large $N$ reduces sensitivity to load changes at the motor; tune gains against effective inertia seen at actuator.

  _Control Summary (for psets)_

  - $tau = J^T f$ maps Cartesian force to joint torques; OSC shapes end-effector dynamics.
  - Stiffness control needs gravity feedforward; smaller $K_p$ increases compliance for uncertainty.
  - Direct force control enables pivoting/insertions but must honor friction cones.
  - Velocity plant (integrator): P for setpoint; PI for constant-velocity tracking; add feedforward of desired velocity.

  _Implementation Recipes_

  == #fa-icon("bolt") InverseDynamicsController

  1) Compute $ddot(q)_(d) := ddot(q)_("ref") + K_p (q_("ref") - q) + K_d (dot(q)_("ref") - dot(q))$.\
  2) Command $u <- M ddot(q)_(d) + C dot(q) - tau_g$.

  == #fa-icon("vector-square") Cartesian Stiffness

  1) Pick frame $E$ at the intended contact.\
  2) Compute $p^E, v^E, J$.\
  3) Choose $K_p, K_d$ (optionally using $M_E$ shaping).\
  4) Compute $attach(f, tl: B, tr: E)_(u) <- -attach(f, tl: B, tr: E)_(g) + K_p (p^(E_d) - p^E) + K_d (dot(p)^(E_d) - dot(p)^E)$, then $u <- J^T attach(f, tl: B, tr: E)_(u)$.

  _Practical iiwa Notes_

  - Cannot send desired joint velocities; firmware differentiates positions (adds small delay) to preserve passivity guarantees.
  - Cartesian impedance mode exists but switching modes/frames requires stopping; commonly stay in joint impedance and use $tau_("ff")$ for Cartesian force cues.
  - Gravity comp uses configured tool inertia; updates not applied online when grasp changes.

  _Peg-in-Hole & Compliance_

  - Avoid jamming with appropriate stiffness about the remote contact center.
  - Remote-Centered Compliance (RCC) devices realize this mechanically (infinite bandwidth, no sensing).
]

// #gatbox(
//   title: [#fa-icon("diagram-project") Ch. 09: Object Detection and Segmentation],
// )[
//   _Detection/segmentation complements geometric pose estimation_

//   == #fa-icon("lightbulb") Big Picture

//   *Limits of geometry:* Accurate but local-minima-prone; struggles in clutter and with many classes.

//   *Deep learning complement:* Large datasets enable global cues: detect objects, segment pixels/points, give coarse pose for geometric refinement.

//   _Datasets and Labels_

//   == #fa-icon("image") ImageNet → Detection

//   *Image-level:* Presence/absence per class.

//   *Object-level:* Bounding boxes + class; fuels detectors.

//   == #fa-icon("vector-square") COCO → Instance Segmentation

//   *Pixel-wise instance masks:* Distinguish instances within class; ideal for picking single item from bin.

//   *Semantic vs instance:* Semantic = per-class mask; Instance = per-object mask (+ class id).

//   _Transfer Learning & Fine-Tuning_

//   == #fa-icon("layer-group") Architecture Pattern

//   *Backbone:* Feature extractor pre-trained on large corpus (ImageNet/COCO).

//   *Head:* Task-specific predictor (classes, boxes, masks). Replace head and fine-tune with small, domain data.

//   _Labeling for Robotics_

//   == #fa-icon("ruler") LabelFusion Workflow

//   *Dense recon:* Merge RGB-D via ElasticFusion to a global point cloud; localize camera.

//   *Fast pose seeding:* Click 3 model ↔ 3 scene points; refine with ICP; render CAD mask into all source frames → pixel-perfect multi-image labels.

//   == #fa-icon("computer") Synthetic Labels (Drake)

//   *RgbdSensor label_image:* Simulator outputs per-pixel instance ids aligned with RGB; unlimited perfectly labeled data for training.

//   *Why synthetic works:* No human noise; at scale can surpass real-data labels when evaluating on real images if domain gap is managed.

//   _Detection & Segmentation Pipelines_

//   == #fa-icon("border-all") Region-Based Detection

//   *R-CNN family:* Region proposals → classify/refine boxes; Fast/Faster R-CNN learn proposals.

//   == #fa-icon("grid-2") Semantic Segmentation

//   *FCN:* Image → image; per-pixel class outputs.

//   == #fa-icon("masks-theater") Instance Segmentation (Mask R-CNN)

//   *Heads in parallel:* Box/class head + mask head; keep masks for top detections.

//   *Tooling:* Prefer torchvision variant for simplicity; Detectron2 often higher performance.

//   _Core Formulas: IoU, AP, NMS_

//   == #fa-icon("ruler-vertical") Intersection-over-Union (IoU)

//   $
//     "IoU"(B_p, B_g) := "area"("intersect"(B_p, B_g)) / "area"("union"(B_p, B_g))
//   $

//   == #fa-icon("chart-line") Precision/Recall and AP

//   $ "Precision" := "TP"/("TP" + "FP"); quad "Recall" := "TP"/("TP" + "FN") $

//   *AP:* Area under Precision–Recall curve. *COCO mAP@[.5:.95]:* Mean AP averaged over classes and IoU thresholds in (0.50, 0.55, ..., 0.95).

//   == #fa-icon("filter") Non-Maximum Suppression (NMS)

//   *Greedy:* Sort boxes by score; keep a box, remove lower-scored boxes with $"IoU" > tau$ (typical $tau = 0.5$).

//   _RPN: Anchors, Targets, Loss_

//   == #fa-icon("border-none") Anchors

//   *Grid anchors:* Multiple scales/aspect ratios per feature map location (often via FPN). Typical: scales in (128, 256, 512), ratios in (1:2, 1:1, 2:1).

//   == #fa-icon("bullseye") Regression Targets

//   Let anchor $a := (x_a, y_a, w_a, h_a)$, ground-truth $g := (x, y, w, h)$.

//   $ t_x := (x - x_a)/w_a; quad t_y := (y - y_a)/h_a; quad t_w := ln(w/w_a); quad t_h := ln(h/h_a) $

//   == #fa-icon("scale-balanced") Loss (objectness + box)

//   $ L_"rpn" := L_"cls"(p, p^*) + lambda sum_(k in (x, y, w, h)) "smoothL1"(t_k - t_k^*) $

//   where $p$ is objectness, $p^* :in Set(0, 1)$. The smoothL1 function equals $0.5 u^2$ for $abs(u) < 1$ and $abs(u) - 0.5$ otherwise.

//   _ROIAlign and Heads_

//   == #fa-icon("crop") ROIAlign

//   *Idea:* Avoid quantization (ROI Pool). For each bin in the fixed grid (e.g., $7 times 7$ or $14 times 14$), bilinearly sample exact floating-point coordinates from the feature map.

//   == #fa-icon("layer-group") Detection Head

//   *Outputs:* class logits $c$, refined boxes $b$ (class-agnostic or per-class).

//   *Loss:* $ L_"det" := L_"cls"(c, c^*) + beta sum "smoothL1"(b - b^*) $.

//   == #fa-icon("masks-theater") Mask Head

//   *Per-class mask logits:* $M^(c) :in RR^(m times m)$ (only $c^*$ used in loss).

//   *Loss:* $ L_"mask" := (1/m^2) sum_(u, v) "BCE"("pred"_(u v), "target"_(u v)) $ where $ "pred"_(u v) := M_(u v)^(c^*) $, $ "target"_(u v) := M_(u v)_"gt"^(c^*) $, and $ "BCE"(a, y) := - y ln(a) - (1 - y) ln(1 - a) $.

//   == #fa-icon("equals") Total Loss

//   $ L := L_"rpn" + L_"det" + L_"mask" $.

//   _Training Details that Matter_

//   == #fa-icon("arrows-up-to-line") Anchor/Proposal Labeling

//   *Positives:* $"IoU" >= t_+$ (e.g., $0.7$); *Negatives:* $"IoU" <= t_-$ (e.g., $0.3$). Sample balanced mini-batches to combat class imbalance.

//   == #fa-icon("shuffle") Augmentation

//   *Geometric:* flips, scales, random crop (apply to boxes/masks consistently). *Photometric:* color jitter; keep masks intact.

//   == #fa-icon("gauge") Inference Thresholds

//   *Scores:* per-class score threshold (e.g., $> 0.5$); run NMS per class; select top-$k$.

//   _From 2D Masks to 3D Geometry_

//   == #fa-icon("image") RGB-D Backprojection

//   For pixel $(u, v)$ with depth $z$ and intrinsics $K$:

//   $ X_C := K^-1 dot [u, v, 1]^T dot z $

//   Filter to the chosen instance's mask to obtain a clean object point cloud.

//   == #fa-icon("cube") Pose Refinement (ICP)

//   Initialize from detection (e.g., class prior) or coarse estimate; run ICP on masked cloud to minimize point-to-model distance.

//   == #fa-icon("hand") Grasping

//   Compute antipodal grasps on filtered cloud; higher success vs full-scene. For suction, fit patch normals on the masked surface only.

//   == #fa-icon("wand-magic-sparkles") Training & Inference Notes

//   - *Head swap:* Replace classifier/box/mask heads; init randomly; warm up with frozen backbone.
//   - *Label mapping:* Align class ids; background = 0; instance ids map to class ids.
//   - *Checkpointing:* Save each epoch; track mAP and mask AP.
//   - *Data:* 10k+ synthetic via Drake (RGB + label_image + metadata), or provided dataset with aligned classes.
//   - *Model:* Replace head; freeze then unfreeze; checkpoint early to avoid timeouts.

//   _Foundation-Scale Segmentation_

//   == #fa-icon("crop-simple") Segment Anything (SAM)

//   *SA-1B:* Massive image/mask dataset built with model-in-the-loop labeling; strong zero-shot masks, then adapt to robotics with light finetuning or prompting.

//   _Using Segmentation in Manipulation_

//   == #fa-icon("fill-drip") Mask → Point Cloud

//   *Filter cloud:* Keep points whose pixels belong to chosen instance; remove clutter.

//   == #fa-icon("compass-drafting") Pose / Grasping

//   *Geometric pose:* Run ICP on segmented cloud to refine 6D pose if needed.

//   *Antipodal grasps:* Compute on filtered cloud; higher success vs full-scene grasps.

//   _Exam-Oriented Reminders_

//   == Differences

//   *Semantic vs Instance:* Class mask vs per-object masks.

//   *Detection vs Segmentation:* Boxes (coarse) vs pixel masks (precise geometry).

//   == Why segment first?

//   *Reduces distractors, improves ICP/grasp selection, stabilizes planning pipelines in clutter.*

//   == Fine-tuning steps

//   *Swap head → align labels → freeze-backbone warmup → unfreeze & train → validate masks and boxes.*
// ]

#colbreak()
#gatbox(
  title: [#fa-icon("diagram-project") Drake Program Outline & Core APIs],
)[
  _End-to-end skeleton_

  - Setup/Diagram/Sim: `StartMeshcat()` → `LoadScenario/MakeHardwareStation` → `DiagramBuilder` add/connect → `Build` → `Simulator.AdvanceTo(T)`.

  _Core APIs_

  - Types/frames: `RigidTransform`, `RotationMatrix`, `EvalBodyPoseInWorld`, `GetFrameByName`.
  - Traj: `PiecewisePose.MakeLinear` → `.MakeDerivative()` for `V_G`; scalars via `PiecewisePolynomial.*Hold`; feed with `TrajectorySource`; integrate with `Integrator(7)`.
  - Controllers: Diff-IK `v = pinv(J) V_G` (from `CalcJacobianSpatialVelocity`); also PD/Impedance, InverseDynamics.
  - Station ports: in `iiwa.position`, `wsg.position`; out `iiwa.position_measured`, camera/point-cloud ports.
  - Gotchas: choose `kV` vs `kQDot` consistently; initialize integrator with current `q`; time WSG open/close around grasp.

  _PseudoInverseController (minimal pattern)_

  - *Ports:* `V_WG: RR^6` ; `iiwa.position: RR^7` → `iiwa.velocity: RR^7`.
  - *Compute:* set plant `q`; `J <- CalcJacobianSpatialVelocity(..., kQDot, G, 0, W, W)[:,0:7]`; `v <- pinv(J) V_G` (damped if needed); output `v`.
  - *Note:* If state uses generalized velocity `v`, use `kV` and map `qdot <-> v`.
  - *Slice dofs:* Use the IIWA block only (e.g., `[:, 0:7]`), or slice via `iiwa_joint_1.velocity_start()` to `iiwa_joint_7.velocity_start()+1`.

  _Traj recipe_

  - *Poses:* `pose_traj <- PiecewisePose.MakeLinear(t, X_WG)` → `traj_V_G <- pose_traj.MakeDerivative()`.
  - *WSG:* `traj_wsg <- PiecewisePolynomial.FirstOrderHold(t, fingers)`.
  - *Sources:* `V_G_source`, `wsg_source` from the above.
  - *Shapes/time:* `fingers` is `1xL`; times `t` must be strictly increasing.

  _Builder wiring_

  - Add `station`; get `plant`. Add `controller`, `integrator`.
  - Connect: `V_G_source -> controller`, `controller -> integrator -> station.iiwa.position`.
  - Feedback: `station.iiwa.position_measured -> controller.iiwa.position`; `wsg_source -> station.wsg.position`.
  - Init: `integrator.set_integral_value(current_q)`.
  - *Get current_q:* `ctx <- station.CreateDefaultContext()`; `plant_ctx <- plant.GetMyContextFromRoot(ctx)`; `current_q <- plant.GetPositions(plant_ctx, plant.GetModelInstanceByName("iiwa"))`.

  _HardwareStation quick notes_

  - *Scenario:* YAML adds models/welds/drivers; `LoadScenario` → `MakeHardwareStation(meshcat)`.
  - *Poses:* objects via `EvalBodyPoseInWorld`; gripper goals `X_WG = X_WO @ X_OG`.
  - *Drivers/weld:* `IiwaDriver(position_only)` with `hand_model_name=wsg`; `SchunkWsgDriver{}`; weld `wsg::body` to `iiwa_link_7` with `Rpy[90,0,90]`.

  _MathProgram / Kinematic TrajOpt_

  - Build: `trajopt <- KinematicTrajectoryOptimization(num_q, N)`; `prog <- trajopt.get_mutable_prog()`.
  - Path constraints: start/goal `PositionConstraint` at `s=0,1`; add joint/vel bounds; zero end velocities.
  - Time: `AddDurationConstraint(Tmin, Tmax)`; costs `AddDurationCost`, `AddPathLengthCost`.
  - Collisions: `MinimumDistanceLowerBoundConstraint` added as path constraints at sampled `s`.
  - Init: warm-start (`trajopt.SetInitialGuess(karate_chop_traj)` or BSpline from RRT); `result <- Solve(prog)`; `traj <- ReconstructTrajectory(result)`; visualize with `PublishPositionTrajectory`.

  _Frames & Jacobian_

  - `CalcJacobianSpatialVelocity(context, wrt, B, p_BoBp_B, A, E)` → 6xN mapping for `V_ABp` expressed in `E`.
  - For world-expressed gripper twist: `A=W`, `E=W`, `B=G`, `p=[0,0,0]`; slice to IIWA dofs.
  - Damped LS near singularities: $v = (J^T J + λ^2 I)^(-1) J^T V$.
]

// TODO: Read grasp sampling portion 5.3 onward
// And PS 04 Q5 https://www.gradescope.com/courses/1074744/assignments/7090415/submissions/new
