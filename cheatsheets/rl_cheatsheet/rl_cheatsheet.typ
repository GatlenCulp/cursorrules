#import "../preamble.typ": *
// This is the final cheatsheet

#show: cheatsheet.with(
  title: "Midterm",
  date: datetime(year: 2025, month: 11, day: 2),
)

#let gatbox = gatbox.with(color-cycle: true)

= PART 0: Mathematical Foundations

#gatbox(
  title: [#fa-icon("calculator") Mathematical Foundations],
)[
  _Background material used throughout course_

  == #fa-icon("square-root-variable") Linear Algebra

  *Least Squares:* $vb(x)^* := (A^T A)^(-1) A^T vb(b) = A^+ vb(b)$ where $A^+$ is pseudoinverse. *Matrix Inversion:* $(I + U V)^(-1) = I - U (I + V U)^(-1) V$

  *Gradients:* $grad_(vb(x)) (vb(x)^T A vb(x)) = (A + A^T) vb(x); quad grad_(vb(x)) (vb(u)^T A vb(x)) = A^T vb(u)$

  // *Matrix Derivatives:*
  // $
  //   pdv(vb(a)^T X^(-1) vb(b), X) = - X^(-top) vb(a) vb(b)^T X^(-top);
  //   quad pdv(vb(x)^T vb(a), vb(x)) = vb(a);
  //   quad pdv(ln(abs(det(X))), X) = X^(-top)
  // $
  // $
  //   pdv((vb(x) - vb(s))^T W (vb(x) - vb(s)), vb(x)) = 2 W (vb(x) - vb(s));
  //   quad pdv((vb(x) - vb(s))^T W (vb(x) - vb(s)), vb(s)) = -2 W (vb(x) - vb(s))
  // $

  == #fa-icon("dice") Probability

  *Variance ($X, Y$ indep):* $"Var"[a X + b Y] = a^2 "Var"[X] + b^2 "Var"[Y]$. *Telescoping:* $V(s_0) = sum_(t=0)^T gamma^t (V(s_t) - gamma V(s_(t+1)))$. *Jensen:* For convex $f$: $f(EE[X]) <= EE[f(X)]$

  == #fa-icon("link") Fixed Point Theory

  *Banach Theorem:* If (A) $(X, d)$ is a complete metric space, (B) $cal(T)$ is a contraction $d(cal(T) x, cal(T) y) <= gamma d(x, y)$ for $gamma in (0,1)$, then unique $x^*$ exists with $cal(T) x^* = x^*$ and $d(x_t, x^*) <= gamma^t/(1-gamma) d(x_1, x_0)$

  // == Normal Distributions
  // *Affine:* $a X + b tilde cal(N) (a mu + b, a^2 sigma^2)$; $A vb(X) + vb(b) tilde cal(N) (A vb(mu) + vb(b), A Sigma A^T)$
  //
  // *Sum (indep):* $X + Y tilde cal(N) (mu_1 + mu_2, sigma_1^2 + sigma_2^2)$
  //
  // *Conditionals:* $mu_(x | y) = mu_x + Sigma_(x y) Sigma_(y y)^-1 (vb(y) - mu_y); quad Sigma_(x | y) = Sigma_(x x) - Sigma_(x y) Sigma_(y y)^-1 Sigma_(y x)$

]

#gatbox(
  title: [#fa-icon("network-wired") Sequential Decision Making Foundations (L01)],
)[
  _L01: Introduction and MDP formulation_

  == #fa-icon("shuffle") Markov Process Tests

  *Projections:* Fail unless eliminated dim doesn't affect future. *Sums:* Fail when different states → same sum. *Subsampling:* Preserves. *Augmentation:* Preserves if all deps captured. *State Augmentation:* Include all lagged states/actions: $overline(S)_(t+1) = B overline(S)_t + C a_t + D w_t$

  == #fa-icon("balance-scale") Expectation vs Optimization

  $min_(a in cal(A)) EE[g(W,a)] >= EE[min_(a in cal(A)) g(W,a)]$. Equality when $a = mu(W)$: $min_(mu in Pi) EE[g(W, mu(W))] = EE[min_a g(W,a)]$
]

= PART 1: Dynamic Programming

#gatbox(
  title: [#fa-icon("route") Finite Horizon Dynamic Programming (L02)],
)[
  _L02: Finite horizon DP algorithm_

  == #fa-icon("hourglass-start") Finite Horizon DP

  $J_t^* (s_t) = min_(a_t in cal(A)(s_t)) EE[c_t (s_t, a_t, w_t) + J_(t+1)^* (s_(t+1))] quad J_T^* (s_T) = c_T (s_T)$
]

#gatbox(
  title: [#fa-icon("cubes") Special Structures (L03-05)],
)[
  _L03: DP arguments, inventory problem, optimal stopping_

  == #fa-icon("warehouse") Base-Stock Policy

  $a_t^* := min((overline(s)_t - s_t)^+, B_t)$ (order up to $overline(s)_t$ if below). Arises from concave value function.
][
  _L04-05: Linear quadratic regulator_

  == #fa-icon("wave-square") LQR (Linear Quadratic Regulator)

  *Dynamics:* $vb(x)_(t+1) = A vb(x)_t + B vb(u)_t$; *Cost:* $J = 1/2 vb(x)_T^T Q_T vb(x)_T + 1/2 sum_(t=0)^(T-1) (vb(x)_t^T Q vb(x)_t + vb(u)_t^T R vb(u)_t)$

  === Finite Horizon

  *Policy:* $vb(u)_t^* = -K_t vb(x)_t$ where gain $K_t = (R + B^T P_(t+1) B)^(-1) B^T P_(t+1) A$. *Value:* $V_t^* (x) = 1/2 vb(x)^T P_t vb(x)$

  *Riccati:* $P_t := Q + A^T P_(t+1) A - A^T P_(t+1) B (R + B^T P_(t+1) B)^(-1) B^T P_(t+1) A$ with $P_T = Q_T$

  === Infinite Horizon

  *Policy:* $vb(u)^* = -K vb(x)$ where gain $K = (R + B^T P B)^(-1) B^T P A$. *ARE:* $P = Q + A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A$

  *Stability:* If $(A,B)$ controllable and $(A,Q^(1/2))$ observable, then closed-loop stable. *Lyapunov:* $P = (A - B K)^T P (A - B K) + Q + K^T R K$ shows $vb(x)_t^T P vb(x)_t$ strictly decreases

  == #fa-icon("random") Stochastic LQR

  *Dynamics:* $vb(x)_(t+1) = A_t vb(x)_t + B_t vb(u)_t + vb(w)_t$ with $A_t, B_t, vb(w)_t$ mutually independent. *Policy:* $vb(u)_t^* = L_t vb(x)_t$ where $L_t = -(R_t + EE[B_t^T K_(t+1) B_t])^(-1) EE[B_t^T K_(t+1) A_t]$

  *Riccati:* $K_t = EE[A_t^T K_(t+1) A_t] - EE[A_t^T K_(t+1) B_t] (R_t + EE[B_t^T K_(t+1) B_t])^(-1) EE[B_t^T K_(t+1) A_t] + Q_t$ with $K_T = Q_T$ (expectations around random matrices)

  == #fa-icon("sliders") Controllability

  $(A,B)$ controllable if $cal(C) = mat(B, A B, ..., A^(n-1) B)$ is rank-$n$. Can then place eigenvalues of $A - B K$ arbitrarily.
]

#gatbox(
  title: [#fa-icon("infinity") Infinite Horizon Dynamic Programming (L06-07)],
)[
  _L06-07: Bellman equations, value iteration, policy iteration_

  == #fa-icon("infinity") Bellman Operator

  $cal(T) V(s) := min_a (c(s,a) + gamma sum_(s') p(s' | s,a) V(s'))$\ $cal(T)_mu V(s) := c(s, mu(s)) + gamma sum_(s') p(s' | s, mu(s)) V(s')$

  *Properties:* Monotonicity ($V <= V' => cal(T) V <= cal(T) V'$), Contraction ($norm(cal(T) V - cal(T) V') <= gamma norm(V - V')_oo$), Fixed Point ($cal(T) V^* = V^*$ unique), Convergence ($lim_(k->oo) cal(T)^k V_0 = V^*$)

  *Matrix Form:* $V^pi = R^pi + gamma P^pi V^pi => V^pi = (I - gamma P^pi)^(-1) R^pi$

  == #fa-icon("repeat") Value Iteration

  $V_(k+1) = cal(T) V_k$ converges: $norm(V_k - V^*) <= gamma^k norm(V_0 - V^*)$. *Greedy:* $mu_k (s) in arg min_a (c(s,a) + gamma sum_(s') p(s' | s,a) V_k (s'))$

  == #fa-icon("sync") Policy Iteration

  1. *Eval:* Solve $V^mu = cal(T)_mu V^mu$; 2. *Improve:* $mu' in arg min_a (c(s,a) + gamma sum_(s') p(s' | s,a) V^mu (s'))$; 3. Repeat. Each iteration improves: $V^(mu') <= V^mu$

  == #fa-icon("code-compare") VI vs PI

  *PI:* Fewer iterations, sensible policies, expensive per iteration (solve $abs(cal(S))$ linear eqs), better for small state spaces. *VI:* Simple updates, more iterations, intermediate policies not guaranteed to improve, better for large state spaces

  == #fa-icon("chart-line") Performance Loss Bound

  If $norm(V - V^*) <= epsilon$ and $cal(T) V = cal(T)_mu V$, then $norm(V^mu - V^*) <= (2 gamma epsilon)/(1 - gamma)$ (tight bound)
]

= PART 2: Reinforcement Learning

#gatbox(
  title: [#fa-icon("robot") Model-Free Policy Evaluation (L08, L10)],
)[
  _L08: Monte Carlo, TD, TD(λ) -- Estimating $hat(V)^pi$ (possibly) without model_

  All $V$ here are an alias for $hat(V)^pi$

  == #fa-icon("mountain") Monte Carlo (MC/TD(1))

  *Update:*
  $
    V_(n+1) (s_0) <- (1-eta_(n+1)) V_n (s_0) + eta_(n+1) hat(R)_(n+1) (s_0)
  $
  where $hat(R)_i (s_0) := sum_(t=0)^(T_i) gamma^t r_(t,i)$ (full return)

  *Properties:* High variance, no bias, needs full episode
  //  *MC as TD:* Incremental MC equivalent to:
  // $
  //   V_(n+1) (s_0) <- V_n (s_0) + eta_(n+1)
  //   sum_(t=0)^(T_(n+1)) gamma^t delta_(t,n+1)
  // $


  == #fa-icon("clock") Temporal Difference Methods

  === TD(0)

  *Update:*
  $
    V_(t+1) (s) <- V_t (s) + eta_t delta_t quad "where" quad delta_t := r_t + gamma V_t (s') - V_t (s)
  $

  *Properties:* Bootstrapped one-step, low variance, biased, online

  === TD($lambda$)

  *Backward View (Eligibility Traces):* $forall s: V(s) <- V(s) + eta_t delta_t e_t (s)$ where $e_t (s) := gamma lambda e_(t-1) (s) + bb(1)(s_t = s)$ tracks recently-visited states (more efficient)

  *Forward View:* Mixes $n$-step returns: $G_t^((lambda)) := (1-lambda) sum_(n=1)^(T-t) lambda^(n-1) G_t^((n))$ where $G_t^((n)) := sum_(k=0)^(n-1) gamma^k r_(t+k) + gamma^n V(s_(t+n))$

][
  _L10: Convergence of TD Methods -- Noisy bootstrapped updates work_

  Not obvious that bootstrapping doesn't blow up your system (TD(1) converges more simply).

  == #fa-icon("water") Martingales and Supermartingales

  *Martingale:* Sequence ${M_t}$ w.r.t. $cal(F)_t$ where $EE[abs(M_t)] < oo$ and $EE[M_(t+1) | cal(F)_t] = M_t$ (expected value doesn't change). *MDS:* ${w_t}$ where $EE[w_t | cal(F)_t] = 0$ (zero-mean noise conditional on past)

  *Supermartingale:* $EE[M_(t+1) | cal(F)_t] <= M_t$ (expected value decreases or stays same). Used in convergence proofs to bound error terms.

  == #fa-icon("chart-simple") Stochastic Approximation (SA)

  *Goal:* Find $theta^* = h(theta^*)$ using noisy $H(theta, xi_t)$ where $EE[H(theta, xi_t)] = h(theta)$. *Update:* $theta_(t+1) <- theta_t + eta_t (H(theta_t, xi_t) - theta_t)$

  *Decomposition:* $H(theta_t, xi_t) - theta_t = (h(theta_t) - theta_t) + w_t$ where $w_t$ is MDS. At fixed point: $h(theta^*) = theta^*$

  *TD(0) as SA:* $V_(t+1) <- V_t + eta_t E_t circle.small (cal(T)_pi V_t - V_t + w_t)$ where $E_t$ is one-hot at $s_t$, $w_t := cal(T)_pi^E V_t - cal(T)_pi V_t$

  == #fa-icon("stairs") Robbins-Monro Stepsize
  $
    underbrace(sum_(t=0)^oo eta_t = oo, "Updates non-negligible")
    quad "and" quad
    underbrace(sum_(t=0)^oo eta_t^2 < oo, "Noise diminishes")
    quad (e.g. eta_t := 1/(t+1))
  $

  == #fa-icon("arrows-to-dot") Convergence Insights

  *Max Norm Convergence:* RM stepsizes + unbiased noise + bounded variance ($EE[w_t^2 (s) | cal(F)_t] <= A + B norm(x_t)^2$) + $h$ is max-norm contraction $=>$ $theta_t -> theta^*$

  *Segment Analysis:* Split time into segments where $product_(t=t_i)^(t_(i+1)-1) (1-eta_t) <= 1/2$, then $abs(EE[x_(t_(i+1))]) <= abs(EE[x_(t_i)])/2$ (exponential convergence of bias). *Variance:* $v_(i+1) <= 1/4 v_i + epsilon_i$ where $epsilon_i -> 0$. Both bias and variance → 0

  *Expected Progress:* If $cal(T)_pi$ is $gamma$-contraction: $EE[norm(V_(t+1) - V^pi)_oo | cal(F)_t] <= (1 - eta_t (1-gamma) p_t) norm(V_t - V^pi)_oo$ where $p_t :=$ prob of updating worst state
]

#gatbox(
  title: [#fa-icon("gamepad") Model-Free Control (L09)],
)[
  _L09: Q-learning, SARSA_

  == #fa-icon("arrows-split-up-and-left") SARSA vs Q-Learning

  *SARSA (On-Policy):*
  $
    Q_(t+1) (s_t, a_t) <- Q_t (s_t, a_t) + eta_t [r_t + gamma Q_t (s_(t+1), a_(t+1)) - Q_t (s_t, a_t)]
  $
  where $a_(t+1) tilde pi(dot | s_(t+1))$. Learns about current policy.

  *Q-Learning (Off-Policy):*
  $
    Q_(t+1) (s_t, a_t) <- Q_t (s_t, a_t) + eta_t [r_t + gamma max_(a') Q_t (s_(t+1), a') - Q_t (s_t, a_t)]
  $
  Learns optimal $Q^*$ regardless of behavior policy. Converges to $Q^*$ under: coverage (all $(s,a)$ visited infinitely often), RM stepsizes, bounded rewards.
]

#gatbox(
  title: [#fa-icon("layer-group") Approximate Value-Based RL (L12-13)],
)[
  #link(
    "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1ee1a73f-8474-4134-9a0f-b32f00c5caa9",
  )[_L12: Function approximation, approximate policy evaluation_]
  -- _Parameterized value functions when tabular representation is infeasible or suboptimal._

  == #fa-icon("shapes") Value Approximation/Parameterization

  *Tabular Value:* Independent value for each state (or state-action pair)

  *Approximate/Parameterized Value* ($V_theta$, $Q_theta$ w/ params $theta$):
  - *Must use:* Continuous/large state spaces (robotics, Atari, Go)
  - *Should use:* Related states share features; sample efficiency; generalization to unvisited states
  - Parameterized doesn't necessarily mean approximate (can restrict hypothesis space for efficiency)
  - Convergence guarantees may be lost depending on parameterization

  === *Common Parameterizations* ($V_theta in cal(F)$):

  *Linear:*
  $
    V_theta (s) = theta^T phi.alt(s) quad "where" quad phi.alt : cal(S) -> RR^d
  $
  Features $phi.alt$ are hand-crafted, can be nonlinear (Fourier, RBF, polynomials). Convex optimization, convergence guarantees.

  *Neural Nets:* $V_theta (s) = "NN"_theta (s)$ - nonlinear in features and weights, learned jointly. Powerful but no convergence guarantees.


  == #fa-icon("diagram-project") Approximate Policy Evaluation (Linear)

  With non-tabular values, we make analogs to Model-Free Policy Evaluation methods:

  *Approximate MC/TD(1):* Supervised learning on full returns
  $
    hat(theta)_n <- arg min_theta 1/n sum_(i=1)^n (V_theta (s_i) - R_i)^2 quad "(Offline)" \
    hat(theta)_(i+1) <- hat(theta)_i - alpha_i grad_theta (V_theta (s_i) - R_i)^2 quad "(Online)"
  $
  where $R_i = sum_(t=0)^(T_i) gamma^t r_(t,i)$ (discounted return). Very sample inefficient, no one uses.

  *Approximate TD(0):* Semi-gradient descent (freeze bootstrapped target)
  $
    hat(theta)_(t+1) <- hat(theta)_t + alpha_t delta_t grad_theta V_theta (s_t) |_(theta = hat(theta)_t)
  $
  where $tilde(R)_t := r_t + gamma V_(hat(theta)_t) (s_(t+1))$ and $delta_t := tilde(R)_t - V_hat(theta)_t (s_t)$

  Why "semi-gradient"? We treat $tilde(R)_t$ as constant, ignoring $grad_theta V_theta (s_(t+1))$ term. Online, efficient.

  *LSTD (Least Squares TD):* Model-based, closed-form solution
  - *Key idea:* Find fixed point of $Pi cal(T)_pi$ (projection composed with Bellman operator)
  - *Why:* True $V^pi$ may lie outside function class; seek $V_theta$ such that $Pi cal(T)_pi V_theta = V_theta$
  - *Requirements:* Linear parameterization, stationary distribution $rho^pi$ (or estimate from samples), model/batch samples
  - *Solution:* Solve weighted least squares
  $
    theta^* = arg min_theta sum_s rho^pi (s) (cal(T)_pi V_theta (s) - V_theta (s))^2
  $
  which gives closed-form: $theta^* = (Phi^T D Phi)^(-1) Phi^T D cal(T)_pi V_theta$ where $D = "diag"(rho^pi)$, $Phi$ is feature matrix
  - *Convergence:* Converges to unique fixed point of $Pi cal(T)_pi$

  *Linear TD:* Model-free iterative version of LSTD
  $
    theta_(t+1) <- theta_t + eta_t phi.alt(s_t) (r_t + gamma phi.alt(s_(t+1))^T theta_t - phi.alt(s_t)^T theta_t)
  $
  Can be written as iterative estimation of $A theta = b$:
  $
    A = EE[phi.alt(s) (phi.alt(s) - gamma phi.alt(s'))^T]; quad b = EE[phi.alt(s) r]
  $
  Solution: $theta^* = A^(-1) b$ (same as LSTD). Converges to LSTD fixed point under RM stepsizes and coverage.

  *Trade-off:* LSTD is faster (closed-form) but needs model/batch; Linear TD is slower (iterative) but model-free.

  *Deadly triad:* Function approximation + bootstrapping + off-policy can cause divergence

  // TODO: What is required of this, what is this about possibly diverging in lecture? Divergence of Off-policy Linear TD(0)


  *Performance Loss with Approximation:* If policy $pi$ satisfies:
  $
    cal(T)_pi V_pi (i) >= cal(T) V_pi (i) - delta quad forall i
  $
  Then: $V^* (i) - V_pi (i) <= delta/(1-gamma) quad forall i$

  Shows approximate greedy policies still perform well.

][
  #link(
    "https://mit.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8ce70d69-6680-4903-a4f9-b32f00c5cac2",
  )[_L13: Approximate VI, fitted Q iteration, DQN, DDQN_]

  Although typically used interchangeably -- *Parameterized* (value with parameters) ⊃ *Approximate* (approximation error even at optimum) ⊃ *Fitted* (refers to the value approximation but also the process by which it is updated -- uses supervised learning instead of analytical projection)

  == #fa-icon("arrows-rotate") Approximate Value Iteration (AVI)
  - *Normal VI:* $V_(k+1) <- cal(T) V_k$, contraction implies $V_k -> V^*$
  - *Approximate VI:* VI on any parameterized value function. After each Bellman update: $V_(k+1) <- cal(A) cal(T) V_k$ where generic function approximation operator $cal(A)$ is often $:= arg min_(V in cal(F)) norm(f - V)_oo$, maps any function to the representable class $cal(F)$ (linear basis, kernels, deep nets, ...) ($Pi^rho$ before was an example)
    - Note: This is more of an analytical definition, in practice $cal(F)$ is huge, you can't do the approximation/projection $cal(A)$ directly. That's why we need DQNs or Fitted Q-Iteration. Fitted Q-iteration is offline (fixed dataset). Q-learning is online (continuous interaction)

  *Fitted Q-Iteration (FQI):* Offline supervised learning approach to AQI
  $
    hat(theta)_(i+1) <- hat(theta)_i - alpha_i grad_theta tilde(cal(L)) (s_(1:n), a_(1:n), y_(1:n); theta) \
    = hat(theta)_i - alpha_i sum_t (
      Q_theta (s_t, a_t) - r_t - gamma max_(a') Q_(theta_i) (s_(t+1), a')
    ) grad_theta Q_theta (s_t, a_t)
  $
  where $tilde(cal(L))$ is empirical loss over dataset, $y_t$ is TD-error. Uses target network $Q_(theta_i)$ (frozen during update) for stability.


  *Performance Bound:* After $K$ iterations with greedy policy $pi_K$:
  $
    norm(V^* - V^(pi_K))_oo <= (2 gamma) / ((1 - gamma)^2)
    max_(0 <= k < K) norm(cal(T) V_k - cal(A) cal(T) V_k)_oo
    + (2 gamma^(K+1)) / (1 - gamma) norm(V^* - V_0)_oo
  $

  *Properties:* If $cal(A)$ is $L_oo$-projection, then non-expansive and $cal(A) cal(T)$ is contraction with unique fixed point $tilde(V)$ where $tilde(V) = cal(A) cal(T) tilde(V)$

  == #fa-icon("brain") Deep Q-Networks

  *DQN:* Q-learning (also kind of FQI) with neural nets $Q_theta (s, a)$. Stabilization techniques:
  - *Replay buffer:* Store transitions $(s, a, r, s')$; sample i.i.d. batches to break temporal correlation
  - *Target network:* Frozen $Q_(theta^-)$ updated periodically for consistent targets $y = r + gamma max_(a') Q_(theta^-) (s', a')$
  - Issue: Overestimates $Q$ due to $max_(a') Q(s', a')$ over noise

  *Double DQN:* Decouple action selection (online net) from evaluation (target net):
  $
    y = r + gamma Q_(theta^-) (s', arg max_(a') Q_theta (s', a'))
  $
  Reduces overestimation bias by not using same network for both selection and evaluation.
]

#gatbox(
  title: [#fa-icon("chart-area") Policy Gradient Methods (L14-15)],
)[
  _L14: Policy gradient, variance reduction_
  _Instead of a tabular policy $pi$ with independent states, lets also parameterize it $pi_theta$ for the same reasons we might want to approximate value. Now, policy iteration breaks._

  - MUST use a parameterized policy when ACTION space is continuous or extremely large
  - SHOULD use a parameterized policy when your STATE space is large enough that maintaining separate policy parameters for each state is unwieldy (Can still have large state space and use a linear parameterization of the policy $pi(a | s) = "softmax"((theta^(a))^top phi(s))$ but doing so requires a full parameter vector for each action)
  // Don't necessarily need a parameterized value to have a parameterized value function.

  == #fa-icon("trophy") Policy Gradient Theorem

  - $mu$ is the initial state distribution.
  - $V_mu (pi_theta) = EE_(s_0 ~ mu) [V^(pi_theta) (s_0)]$

  $
    grad_theta V_mu (pi_theta) = EE_(tau tilde P_(mu)^(pi_theta)) [
      R(tau) dot sum_(t=0)^oo grad_theta log pi_theta (a_t | s_t)
    ]
  $
  where $R(tau) := sum_(t=0)^oo gamma^t r(s_t, a_t)$
  // = G_0 as defined down in reinforce.

  *Intuition:* Increase log-probability of actions proportional to their return

  *Causality:* Past rewards don't affect gradient at time $t$, so can replace $R(tau)$ with future returns:
  $
    sum_(t'=t)^oo gamma^(t'-t) r_(t')
  $

  *Actor-Critic Form:*
  $
    grad_theta V_mu (pi_theta) = EE_(tau tilde P)[
      sum_(t=0)^oo gamma^t Q^(pi_theta) (s_t, a_t) dot grad_theta log pi_theta (a_t | s_t)
    ]
  $

  *Discounted State Distribution Form:*
  $
    grad_theta V_mu (pi_theta) = 1/(1-gamma) EE_(s tilde d_(mu)^(pi_theta)) EE_(a tilde pi_theta) [
      Q^(pi_theta) (s,a) dot grad_theta log pi_theta (a | s)
    ]
  $
  where $d_(mu)^(pi_theta) (s) := (1-gamma) sum_(t=0)^oo gamma^t Pr(s_t = s | pi_theta, mu)$

  == #fa-icon("fire") REINFORCE ($grad_theta V_theta$ using MC/TD(0) Policy Evaluation)

  *Algorithm:* Monte Carlo policy gradient
  1. Collect episode $tau$ under $pi_theta$
  2. Compute returns: $G_t := sum_(t'=t)^T gamma^(t'-t) r_(t')$
  3. Update:
  $
    theta <- theta + eta sum_(t=0)^T G_t grad_theta log pi_theta (a_t | s_t)
  $

  *Properties:* On-policy, high variance, unbiased gradient estimates

  *Variance Reduction with Baseline:*
  $
    theta <- theta + eta sum_(t=0)^T (G_t - b(s_t)) grad_theta log pi_theta (a_t | s_t)
  $
  Common choice: $b(s_t) := V(s_t)$ (value function baseline)

  Best: Use advantage $A(s_t, a_t) := Q(s_t, a_t) - V(s_t)$ for variance reduction

][
  _L15: Actor-critic methods -- Compatible function approximation, A2C, A3C, DDPG, SAC_

  == #fa-icon("handshake") Actor-Critic Methods

  *Idea:* Combine value-based and policy-based approaches
  - *Actor:* Policy network $pi_theta$ that takes actions
  - *Critic:* Value function $V_phi$ or $Q_phi$ that estimates values to reduce variance

  // The critic doesn't NECESSARILY have to be parameterized, could be tabular. But using parameters and sharing parameters makes them more generalizable and sample efficient.

  // In general -- use TD methods/bootstrapping to do online learning

  // offline learning is a batched set of episodes
  // and online learning is after each episode
  // What are bootstrapped methods that learn after each step within an episode called?
  // Answer -- in RL online typically means both step-by-step updates or episode-by-episode updates (confusingly?) Typically online just means learning as data is collected.

  // TD ⊂ bootstrapping


  === A2C (Advantage Actor-Critic)

  *Update:* Synchronous version of A3C
  $
    theta <- theta + eta grad_theta log pi_theta (a_t | s_t) hat(A)_t
  $
  where advantage $hat(A)_t := sum_(i=0)^(n-1) gamma^i r_(t+i) + gamma^n V_phi (s_(t+n)) - V_phi (s_t)$ ($n$-step TD)

  *Properties:* On-policy, synchronous (waits for all workers), stable training

  // NOTE: hat(A)_t is an n-step TD estimate of the advantage. If n=1 then this IS TD(0) ()

  === A3C (Asynchronous Advantage Actor-Critic)

  *Key difference:* Multiple workers train asynchronously on different environment copies, update shared parameters without waiting

  *Benefits:* Decorrelates data (like replay buffer), faster wall-clock time, more exploration diversity

  === DDPG (Deep Deterministic Policy Gradient)

  *For:* Continuous action spaces. *Policy:* Deterministic $mu_theta (s)$

  *Actor update:* $grad_theta EE[Q_phi (s, mu_theta (s))] = EE[grad_theta mu_theta (s) grad_a Q_phi (s,a) |_(a = mu_theta (s))]$

  *Critic update:* TD learning for $Q_phi$ with target networks and replay buffer (like DQN)

  *Exploration:* Add noise $a_t = mu_theta (s_t) + epsilon_t$ where $epsilon_t tilde cal(N) (0, sigma^2)$ (or Ornstein-Uhlenbeck)

  === TD3 (Twin Delayed DDPG)

  *Improvements over DDPG:*
  1. *Twin Q-networks:* Use $min(Q_(phi_1), Q_(phi_2))$ to reduce overestimation
  2. *Delayed policy:* Update actor less frequently than critic
  3. *Target smoothing:* Add noise to target actions $tilde(a) = mu_(theta') (s') + epsilon$ where $epsilon tilde "clip"(cal(N) (0, sigma^2), -c, c)$

  === SAC (Soft Actor-Critic)

  *Framework:* Entropy-regularized RL (see L16) with actor-critic

  *Objective:* $max EE[sum_t gamma^t (r_t - alpha log pi_theta (a_t | s_t))]$ where $alpha$ is temperature

  *Updates:* Actor maximizes $EE[Q_phi (s,a) - alpha log pi_theta (a | s)]$; critic learns soft Q-function

  *Properties:* Off-policy, stochastic policy, automatic exploration, stable
]


#gatbox(
  title: [#fa-icon("fire-flame-curved") Advanced Policy Methods (L16)],
)[
  _L16: Conservative policy iteration, NRG, TRPO, PPO_

  == #fa-icon("shield-halved") Conservative Policy Iteration

  *Problem:* Large policy updates can cause performance collapse

  *Idea:* Constrain policy updates to stay close to current policy using KL divergence

  *General form:* $max_theta EE[...] quad "s.t." quad EE_(s tilde d^pi) [KL(pi_theta' (dot | s), pi_theta (dot | s))] <= delta$

  == #fa-icon("shield") TRPO (Trust Region Policy Optimization)

  *Objective:* Maximize surrogate advantage while constraining KL divergence
  $
    max_theta EE_(s,a tilde pi_"old") [(pi_theta (a | s))/(pi_"old" (a | s)) A^(pi_"old") (s,a)]
  $
  subject to: $EE_(s tilde d^(pi_"old")) [KL(pi_"old" (dot | s), pi_theta (dot | s))] <= delta$

  *Solution:* Uses conjugate gradient and line search to approximately solve constrained optimization

  *Properties:* Monotonic improvement guarantee, but computationally expensive

  == #fa-icon("scissors") PPO (Proximal Policy Optimization)

  *Idea:* Simpler alternative to TRPO - clip objective instead of hard constraint

  *PPO-Clip:*
  $
    cal(L)^"CLIP" (theta) = EE_t [min(r_t (theta) hat(A)_t, "clip"(r_t (theta), 1-epsilon, 1+epsilon) hat(A)_t)]
  $
  where $r_t (theta) := (pi_theta (a_t | s_t))/(pi_"old" (a_t | s_t))$ is probability ratio

  *Clipping:* Prevents ratio from moving outside $[1-epsilon, 1+epsilon]$ (typically $epsilon = 0.2$)

  *PPO-Penalty:* Add KL penalty to objective instead of clipping
  $
    cal(L)^"KL" (theta) = EE_t [r_t (theta) hat(A)_t - beta KL(pi_"old" (dot | s_t), pi_theta (dot | s_t))]
  $

  *Properties:* Simple to implement, sample efficient, widely used (default RL algorithm for many applications)

  == #fa-icon("temperature-high") Entropy-Regularized RL

  *Framework:* Add entropy bonus to encourage exploration

  *Boltzmann Policy:* $pi_Q^cal(B) (a | s) := refPol(a | s) dot (exp(Q(s,a)/tau))/(EE_(a tilde refPol) [exp(Q(s,a)/tau)])$. *Soft Value:* $V_(pi_Q^cal(B)) (s) = tau log EE_(a tilde refPol) [exp(Q(s,a)/tau)]$

  == #fa-icon("snowflake") Soft Q-Learning

  Learn $Q$ function for entropy-regularized objective. Alternate: (1) *Policy improvement:* $pi <- pi_Q^cal(B)$; (2) *Policy evaluation:* Update $Q$ toward soft Bellman target

  *Single-Sample Q-Update:* Minimize $1/2 norm(Q_theta (s_t, a_t) - y_t)^2$ where $y_t := r_t + gamma V_(pi_Q^cal(B)) (s_(t+1))$ (semi-gradient). Uses soft value instead of $max$ in Q-learning
]

= PART 3: Special Topics

#gatbox(
  title: [#fa-icon("dice") Multi-Armed Bandits (L17)],
)[
  _L17: Multi-arm bandits -- Exploration vs exploitation with horizon = 1_

  == #fa-icon("circle-info") Setup & Regret

  At each round $t$, select $a_t in cal(A)$, receive $r_t (a_t)$ #footnote[The $t$ subscript on $r$ is only meant to indicate the actual reward received at that time(?) Not that the reward changes with time.]

  *Notation:*#footnote[Delta is called the "Gap"]
  $
    mu(a) := EE[r_t (a)]; quad
    a^* := arg max_a mu(a); quad
    T_n (a) := sum_(t=1)^n bb(1){a_t = a}
  $
  $
    A := |cal(A)|; quad
    Delta(a) := mu(a^*) - mu(a)
  $

  *Regret:*
  $
    R_n = max_a EE[sum_(t=1)^n r_t (a)] - EE[sum_(t=1)^n r_t (a_t)] = sum_(a != a^*) EE[T_n (a)] Delta(a)
  $

  == #fa-icon("search") Explore-then-Commit

  *Algorithm:*
  1. *Explore:* Pull each arm $K$ times (total $tau = K A$ rounds)
  2. *Commit:* Choose $hat(a)^* = arg max_a hat(mu)_tau (a)$ and pull for remaining $n - tau$ rounds

  where $hat(mu)_tau (a) = 1/K sum_(t: a_t = a)^tau r_t (a)$ is empirical mean

  *Regret:*  #footnote[
    How to show exploiation error:
    - Let confidence radius
      $ r(a) = sqrt(2 ln(n) slash K) $
    // What is this?
    -
  ]
  $
    R_n <= underbrace(tau, "exploration cost") + underbrace(cal(O)(sqrt(A log n) / tau n), "exploitation error")
  $


  == #fa-icon("random") $epsilon$-Greedy

  *Algorithm:* At each round $t$:
  - With probability $epsilon_t$: explore (pick random action)
  - With probability $1 - epsilon_t$: exploit (pick $arg max_a hat(mu)_t (a)$)

  *Online Regret:* $R_t <= tilde(cal(O))(t^(2/3))$ with $epsilon_t = t^(-1/3) (A log t)^(1/3)$

  == #fa-icon("arrow-up") Upper Confidence Bound (UCB)

  *Idea:* Optimism under uncertainty - construct upper confidence bound $B_t (a)$ for each action and pick action with highest bound

  *Algorithm:*
  1. Compute estimates: $hat(mu)_t (a) = 1/(T_t (a)) sum_(s=1)^t r_s bb(1){a_s = a}$
  2. Evaluate uncertainty: $abs(hat(mu)_t (a) - mu(a)) <= sqrt((log 2/delta_t)/(2 T_t (a)))$ w.h.p.
  3. Optimism (exploration bonus):
  $
    B_t (a) = hat(mu)_t (a) + underbrace(rho, "tuning param") sqrt((log 2/delta_t)/(2 T_t (a)))
  $
  4. Select: $a_t = arg max_a B_t (a)$

  Actions with high uncertainty get larger exploration bonus

  *Regret:* With $rho = 1$, $delta_t = 1/t$:
  $
    R_n = cal(O)(sum_(a != a^*) (log n)/Delta(a))
  $
  (near-optimal, almost matches lower bounds)

  *Tuning $rho$:* $rho < 1$ → polynomial regret; $rho >= 1$ → logarithmic regret. Larger $rho$ → more exploration.

  == #fa-icon("star") Recommendation System (RS)
  - Similar to linear value approximation, when the action space is extremely large, parameterize actions with features
  - Contextual bandits (have a linear context variable, e.g. on a user)

  == #fa-icon("chart-line") Concentration Inequalities

  *Hoeffding's Inequality:* Sample mean concentrates around true mean exponentially fast. For $i in {1:n}$, let $X_i in [a, b]$ be independent RVs with mean $mu$. Then
  $
    Pr(
      mark(abs(overline(X)_n - mu), tag: #<deviation>, color: #red.darken(40%))
      > mark(epsilon, tag: #<accuracy>, color: #orange)
    )
    <= mark(
      2 exp(- (2 n epsilon^2) / (b - a)^2),
      tag: #<confidence>, color: #green.darken(40%)
    )
    // "deviation threshold"
    #annot(<deviation>, pos: top)[Deviation]
    #annot(<accuracy>)[Accuracy]
    #annot(<confidence>)[Confidence (exp in $n$)]
  $
  where $overline(X)_n := 1/n sum_(i=1)^n X_i$. Prob of large deviation (> $epsilon$) decays exponentially with $n$.

  *Chernoff-Hoeffding:* Equivalent form, setting $delta$ instead of $epsilon$. showing confidence radius. For fixed failure prob $delta$, estimated means concentrate in radius shrinking as $sqrt(n)$:
  $
    Pr(
      mark(abs(overline(X)_n - mu), tag: #<deviation>, color: #red.darken(40%))
      >
      mark(
        (b - a)
        sqrt((log 2/delta) / (2 n)),
        tag: #<accuracy>,
        color: #orange
      )
    ) <= mark(delta, tag: #<confidence>, color: #green.darken(40%))
    #annot(<accuracy>)[Confidence Radius\ (shrinks with $sqrt(n)$)]
    #annot(<deviation>, pos: top)[Deviation]
    #annot(<confidence>)[Fixed failure prob]
  $

  Both equivalent; Hoeffding shows tail probability, Chernoff-Hoeffding shows confidence radius

  *Hack*: When you only have a one-sided tail, you can decompose the absolute value into one-sided tails

  //   What you know (two-sided Hoeffding):
  // Pr⁡(∣X‾n−μ∣>ϵ)≤2exp⁡(−2nϵ2(b−a)2)\Pr(|\overline{X}_n - \mu| > \epsilon) \leq 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)Pr(∣Xn​−μ∣>ϵ)≤2exp(−(b−a)22nϵ2​)
  // Decompose into one-sided tails:
  // Pr⁡(∣X‾n−μ∣>ϵ)=Pr⁡(X‾n−μ>ϵ)+Pr⁡(X‾n−μ<−ϵ)\Pr(|\overline{X}_n - \mu| > \epsilon) = \Pr(\overline{X}_n - \mu > \epsilon) + \Pr(\overline{X}_n - \mu < -\epsilon)Pr(∣Xn​−μ∣>ϵ)=Pr(Xn​−μ>ϵ)+Pr(Xn​−μ<−ϵ)
]

= Post-Class Notes

== Hierarchical Reinforcement Learning (HRL)

=== HRL Overview

*Promises of HRL* #link("https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/")[[Link]]
+ Long-term credit assignment (faster learning and better generalization)
+ Structured exploration (sub-policies not primitive actions)
+ Transfer learning (hierarchy = different knowledge = better transfer)




Learn multiple policies operating at different levels of abstraction. Multiple frameworks for HRL. One involves thinking of them as options \@hrl-options. Another involves thinking about them temporally \@hrl-temporal. Others.

// Worse link"https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/"

*Options Framework*: Option contains three components (better explained in link, but did not read that first) <hrl-options>
- Initiation Set: When option can be invoked
- Policy: Actions to be taken
- Termination Condition: When option should end
- Option can be primitive (meaning base-level actions.)
+ *Subgoal Discovery*: Idk somehow helps with this
+ *Reward Shaping*: Assign rewards at different levels of the hierarchy.
#link(
  "https://www.researchgate.net/profile/Shengbo-Li-2/publication/338852763/figure/fig1/AS:852158580797441@1580181846660/Hierarchical-RL-for-self-driving-decision-making_W640.jpg",
)[Diagram]

*Temporal Framework*: Macro-actions consisting of sequences of lower-level actions (primitive action is lowest level). #link("https://thegradient.pub/content/images/2019/03/image52.png")[[Image]]

=== Hierarchical DQN (H-DQN)


Two levels:
1. High-Level Controller:
  #footnote[Aka Meta-Controller]
  Selects subgoals.
2. Low-level Controller

== Model-Specific

- *Dreamer*: Model-based RL
- *PPO*: Policy-based RL (Actor-Critic)
- *Rainbow*: Value-based RL (DQN-Enhanced)
- *Plan2Explore*: Model-based Exploration (Intrinsic Reward, often Plan2Explore-DreamerV{X})
- *RND*: Exploration Bonus (Intrinsic Reward Signal Strategy, often RND-PPO)


== Research Techniques

- *Ablation*: Systematically remove or modify components of a model to assess their individual contributions to overall performance.
