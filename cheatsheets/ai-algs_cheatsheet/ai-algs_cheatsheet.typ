#import "@preview/lovelace:0.3.0": pseudocode-list
#set page(paper: "us-letter", margin: 1cm, columns: 3)
#set text(size: 7pt)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  set align(center)
  it
}
#show heading.where(level: 2): it => {
  set align(center)
  set text(size: 8pt)
  it
}

#show heading.where(level: 3): it => {
  set align(center)
  set text(size: 7pt, fill: blue)
  it
}

#let st = $"s.t."$

// Math Control
#let If = $#strong("if")$
#let Then = $#strong("then")$
#let Else = $#strong("else")$

#let llm = text.with(blue.darken(40%))

#heading(level: 1, numbering: none)[6.4110 Cheat Sheet \ (AI Algorithms)]
#align(center)[
  Gatlen Culp (gculp\@mit.edu)
  #sym.dot May 2025
]

#llm[
  = Probability & Bayes Filters

  *Prediction*: $"bel"^-(x_t) = integral p(x_t|x_(t-1),u_(t-1)) dot "bel"(x_(t-1)) dif x_(t-1)$
  - State estimation over time using dynamics model
  - Incorporates control inputs and prior belief

  *Update*: $"bel"(x_t) = "eta" dot p(z_t|x_t) dot "bel"^-(x_t)$
  - Corrects prediction using measurement model
  - $"eta"$ is normalization constant: $1/P(z_t)$

  *Log Odds*: $l_t = l_(t-1) + log(p(z_t|x_t) / p(z_t|bar(x_t)))$
  - Numerically stable for binary states
  - Recover probability: $P(x) = 1 - 1/(1+exp(l))$

  *Filters*:
  - *Kalman Filter (KF)*: Optimal for linear systems with Gaussian noise
  - *Extended Kalman Filter (EKF)*: Linearizes non-linear systems with Jacobians
  - *Unscented Kalman Filter (UKF)*: Propagates carefully chosen sample points
  - *Particle Filter (PF)*: Represents belief with particles, handles multimodal distributions
  - *Histogram Filter*: Discretizes state space, exact but scales poorly

  *Convergence*: PF var $prop 1/N$; resample when $N_"eff" < N/2$
  - *Effective Sample Size (ESS)*: $N_"eff" = 1/sum_i (w_i^2)$
  - Resampling trades variance for bias
]

== Gaussian Formulas
$X ~ cal(N)(mu_x, sigma_x^2)$, $Y ~ cal(N)(mu_y, sigma_y^2)$:

*Sum of Gaussians*
$
  X + Y ~ cal(N)(mu_x + mu_y, sigma_x^2 + sigma_y^2)
$

*Multivariate Gaussian*
$
  (X, Y) ~ cal(N)((mu_x, mu_y),
    mat(
      sigma_x^2, sigma_"xy"^2;
      sigma_"xy"^2, sigma_y^2
    ))
$

*Conditional Gaussian*
$
  P(X|Y=y) = cal(N)(mu_x + (sigma_"xy"^2) / (sigma_y^2)(y - mu_y), sigma_x^2 - (sigma_"xy"^2)^2 / (sigma_y^2))
$


= Graphical Models

*Bayes Net*: Directed acyclic graph representing $ Pr(X_1, ...X_n) = product_i Pr(X_i | "Parents"(X_i)) $

*Markov Network*: Undirected graph with potential functions over cliques

== D-separation
Determines if $X tack.t.double Y | (Z := {Z_1, Z_2, ...})$
#pseudocode-list[
  + Treat edges as undirected, identify paths from $X -> Y$
  + Check if path blocked by $Z$
  + If all paths blocked, then conditionally independent
]

== Markov Blanket (MB)
Shield that makes $X$ independent from all other variables:
$
  "MB(X)" := X."parents" union X."children" union X."spouses"
$
- $X."spouses"$ = parents of $X$'s children
- $P(X | "MB"(X),Y) = P(X | "MB"(X))$

== Variable Elimination \ (Type of Sum-Product on Markov Chains)

In MMs, factor out independent vars by grouping as $gamma_X (Y=y)$ to show providing y isolates the effect of $X$

- You select the ordering
- Requires there be no loops
- Performs exact inference for:
  - Bayes Nets (Directed GMs)
  - MMs *Hidden Markov Models (HMMs)* (Undirected GMs)
  - Factor Graphs
  - Conditional Random Fields


- Complexity: $O(d^{w+1})$ where $w$ = tree-width
- Ordering affects efficiency (min-degree, min-fill heuristics)
- Works for Bayes Nets, Markov models, factor graphs
$
  sum_(d in {0, 1}) [P_(C|D) (c|d) dot P_D (d)] = P_C (c) = gamma_D (c)
$
== Sum-Product (Belief Propagation)

Message-passing algorithm for exact inference on trees:

=== Variable to Factor
$
  mu_(x -> #text(red)[$f$]) (x) = product_(forall g in {#text(green)[$"ne"(x)$] \\ #text(red)[$f$]}) mu_(g -> x) (x)
$
$#text(green)[$"ne"(x)$] := "Neighbors of x"$\
$#text(red)[$f$] = "factor x is passing to"$



=== Factor to Variable
$
  mu_(f -> x) (x)
  = sum_(#text(red)[$cal(X)$] _f \\ x) [ phi.alt_f (#text(red)[$cal(X)$] _f) dot
    product_(#text(green)[$y$] in ("ne"(f) \\ x))
    mu_(#text(green)[$y$] -> f)(#text(green)[$y$]) ]
$
$#text(green)[$y$] = "Incoming vars"$\
$#text(red)[$cal(X)$] _f := "All vars connected to factor" f$ (very confusingly, inside the sum it is referring to the assignments though.)\
$sum_(#text(red)[$cal(X)$] _f \\ x) = "Summation over all assignments of vars" #text(red)[$cal(X)$] _f \\ x$

ie: Marginalize out all other vars

=== Marginal
$
  mu_(f -> x) (x) prop product_(f in "ne"(x)) mu_(f -> x)(x)
$

- Converges to exact solution on trees
- May still work well on loopy graphs (loopy BP)
- Max-product variant finds most likely assignment

= Sampling & Approx Inference

== Bayesian Network Sampling

=== *Ancestral (Forward) Sampling*:
Draw exact i.i.d. samples from joint distribution
#pseudocode-list[
  + order $<-$ topological_sort($cal(G)$) // parents before children
  + for $i = 1 ... M$: // repeat for each sample
    + for $v$ in order: // iterate over all nodes
      + $x_i[v] <- "Sample"(P(v | "pa"(v)=x_i["pa"(v)]))$
  + return ${x_i}_{i=1}^M$
]
- *Estimator*: $hat(E)[f(V)] = 1/M sum_(i=1)^M f(x^((i)))$
- *Pros*: Simple, exact, O(MN) complexity, parallelizable
- *Cons*: Cannot condition on evidence efficiently, needs full CPT access

=== *Rejection Sampling*: Sample from $P(V | E=e)$ by discarding mismatches
#pseudocode-list[
  + accepted $<- []$
  + while $|"accepted"| < M$:
    + $x <- "AncestralSample"(cal(G))$
    + if $x[E] = e$: // accept if consistent with evidence
      + accepted.append(x)
  + return accepted
]
- *Acceptance rate*: $hat(A) = P(E=e)$
- *Expected cost*: O(MN / $hat(A)$)
- *Pros*: Unbiased, exact draws
- *Cons*: Exponential slowdown for rare evidence, wastes computation

=== *Importance Sampling & Likelihood Weighting*: Re-weight samples from easier proposal
- *Generic Importance Sampling*:
$
  E_P[f(X)] = (sum_(i=1)^M w_i f(x^((i)))) / (sum_(i=1)^M w_i)
$
- *Weights*: $w_i = P(x^((i)))/Q(x^((i)))$ where we draw $x^((i))$ from proposal $Q$

- *Likelihood Weighting* (evidence nodes fixed, others sampled forward):
#pseudocode-list[
  + order $<-$ topological_sort($cal(G)$)
  + for $i = 1 ... M$:
    + weight $<- 1$
    + for $v$ in order:
      + if $v in E$:
        + weight $*= P(v=e_v | "pa"(v)=x_i["pa"(v)])$
        + $x_i[v] <- e_v$
      + else:
        + $x_i[v] <- "Sample"(P(v | "pa"(v)=x_i["pa"(v)]))$
    + store $(x_i, "weight")$
  + return ${(x_i,"weight")}_{i=1}^M$
]
- *Weight*: $w(z) = P(z,e)/Q(z) = product_(e_j in E) P(e_j | "pa"(e_j))$
- *Pros*: Uses all draws, works with tiny $P(E=e)$, parallelizable
- *Cons*: High variance if weights spread widely, requires stable numerical handling

*Practical Tips*:
- Pre-compute topological ordering for static networks
- Normalize weights periodically to avoid underflow
- Use log-space accumulation for numerical stability
- Consider adaptive proposals or MCMC when evidence is deep in the graph
- For sequential models, combine LW with systematic resampling (particle filtering)

*Mnemonic*: A-R-I
- Ancestral – *All* nodes forward
- Rejection – *Reject* bad evidence
- Importance – *Importance-weight* good evidence

#llm[
  == Rejection Sampling

  #pseudocode-list[
    + $#strong("function") "REJECTION-SAMPLING"(X, "e", "bn", N)$
    + $#strong("inputs: ") X$, the query variable
      + $"e"$, observed values for variables $"E"$
      + $"bn"$, a Bayesian network
      + $N$, the total number of samples to be generated
    + $#strong("local variables: ") "C"$, a vector of counts for each value of $X$, initially zero
    + $#strong("for ") j = 1 #strong(" to ") N #strong(" do")$
      + $"x" <- "PRIOR-SAMPLE"("bn")$
      + $#strong("if ") "x" " is consistent with " "e" #strong(" then")$
        + $"C"[j] <- "C"[j]+1$ where $x_j$ is the value of $X$ in $"x"$
    + $#strong("return ") "NORMALIZE"("C")$
  ]

  sample from easier distribution, reject if doesn't match evidence
  - Simple but inefficient for unlikely evidence

  // *Likelihood Weighting*: fix evidence variables, weight samples by likelihood
  // - More efficient with evidence but may have high variance
]

== Gibbs Sampling
#pseudocode-list[
  + Initialize all vars $X = (X_1, ..., X_n)$
  + Repeatedly sample $X_i$ from $Pr[X_i | X_(not i)]$ ($X_(not i) := X - {X_i}$)
  + After enough steps, burn-in
]
#llm[
  - Markov blanket simplifies conditional computation: $X_i ~ Pr(X_i|"MB"(X_i))$
]

== Importance Sampling
#pseudocode-list[
  + Choose proposal dist. $Q(X)$ that is easy to sample from
  + Draw samples $x^((1)), ... x^((s))$ from $Q(x)$
  + Compute importance weights $W^((s)) = P(x^((s))) / Q(x^((s)))$
  + Estimate $E[f(X)] approx sum_(i=1)^s (W^((i))/sum_j W^((j))) f(x^((i)))$
]

#llm[
  Sample $x ~ Q$, weight $w=P/Q$; var high if $Q ≉ P$
  - Adaptive importance sampling adjusts proposal during sampling
  - Annealed importance sampling bridges from easy to target distribution
]

#llm[
  == *Markov Chain Monte Carlo (MCMC)*
  - Framework for sampling from complex distributions
  - Constructs Markov chain with target as stationary distribution
  - *Metropolis-Hastings*: Accept new state with probability min(1, P(new)Q(old|new)/P(old)Q(new|old))
  - *Gibbs sampling*: Special case updating one variable at a time
  - Convergence diagnostics: ESS (effective sample size), PSRF < 1.1
]
= Temporal Models

== Hidden Markov Models (HMM)
- Joint: $ Pr(X_(0:T), Z_(1:T)) = Pr(X_0) product_(t=1)^T [Pr(Z_t|X_t) dot Pr(X_t|X_(t-1))] $
- Forward-backward algorithm for smoothing
- Viterbi algorithm for most likely state sequence
- *Smoothing*: Estimating $P(x_t|z_{1:T})$ for some past time $t < T$
  - Uses both past observations (forward pass) and future observations (backward pass)
  - Provides more accurate state estimates than filtering by using all available data
  - Implemented using the forward-backward algorithm combining $alpha_t$ and $beta_t$

=== Viterbi Algorithm
Find most likely state sequence $x_(1:T)^*$ given observations $z_(1:T)$
#pseudocode-list[
  + $delta_1(i) <- P(x_1=i) dot P(z_1|x_1=i)$ for all $i$
  + $psi_1(i) <- 0$ // Backpointers (optional for first timestep)
  + For $t = 2$ to $T$:
    + $delta_t(j) <- max_i [delta_(t-1)(i) dot P(x_t=j|x_(t-1)=i)] dot P(z_t|x_t=j)$
    + $psi_t(j) <- "argmax"_i [delta_(t-1)(i) dot P(x_t=j|x_(t-1)=i)]$
  + $x_T^* <- "argmax"_i delta_T(i)$
  + For $t = T-1$ down to $1$:
    + $x_t^* <- psi_(t+1)(x_(t+1)^*)$
]
- $delta_t(i)$: Probability of most likely path ending in state $i$ at time $t$
- $psi_t(i)$: Backpointer to previous state in most likely path
- Uses log space for numerical stability in practice

#llm[
  == HMM forward
  $
    alpha_t = (A^T dot alpha_(t-1)) dot B(z_t)
  $
  - $alpha_t(i) = Pr(z_{1:t}, x_t=i)$
  - Backward: $beta_t(i) = sum_j A_(i j) B_j(z_{t+1}) beta_{t+1}(j)$
  - Smoothing: $P(x_t|z_{1:T}) prop alpha_t(i) beta_t(i)$
  == Kalman
  $
    "Predition" hat(x)^- & = F hat x \
                     P^- & = F dot P dot F^T + Q \
              "update" K & = P^-H^T(H dot P^- dot H^T+R)^{-1}
  $
  - State update: $hat(x) = hat(x)^- + K(z - H hat(x)^-)$
  - Covariance update: $P = (I - "KH")P^-$
  - Information form efficient for high-dimensional measurements
]

= Graph Search

// #table(
//   columns: 5,
//   [*Alg*], [*Order*], [*Opt?*], [*Compl?*], [*Notes*],
//   [*BFS*], [depth], [Y], [Y], [uniform cost = 1],
//   [*DFS*], [stack], [N], [Y], [O(b^m) time, O(m) mem],
//   [*UCS*], [cost g], [Y], [Y], [non-neg costs],
//   [*Greedy*], [heuristic h], [N], [N], [best-first],
//   [*$A\*$*], [f=g+h], [Y\*], [Y], [h admissible & consistent],
//   [*MCTS*], [UCB], [N], [prob], [Monte Carlo Tree Search; anytime],
// )

*Equivalences*:
- $h=0$ ⇒ A\* ≡ UCS
- $g=0$ ⇒ A\* ≡ Greedy
- cost=1 ⇒ BFS ≡ UCS
- *Admissible*: $h(n) <= "true cost"$
- *Consistent*: $h(n) <= "cost"(n,n') + h(n')$
- A\* with consistent h expands nodes in order of increasing f-value
- Admissible but inconsistent h may require reopening closed nodes

// *Memory*: BFS/UCS/A\* store all nodes; IDA\*/RBFS use O(d) memory
// - IDA\*: iterative deepening on f-cost cutoff
// - RBFS: recursive best-first search with backtracking
// - SMA\*: simplified memory-bounded A\* with node dropping

// *Search Enhancements*:
// - Bidirectional: search from both start and goal
// - MM: meet in the middle heuristic for bidirectional search
// - Beam search: keep only k best nodes at each level

= Classical Planning Heuristics

// *Delete-relax*: $h_"blind" ≤ h_max ≤ h_"add" ≤ h_"ff"$ ; $h_max$, lmcut admissible
// - $h_"blind"$: constant (0 or goal cost)
// - $h_max$: max cost over all subgoals (admissible)
// - $h_"add"$: sum earliest-layer costs in relaxed planning graph
// - $h_"ff"$: greedy solution length in RPG (inadmissible)

// *Landmark cut*: iteratively remove landmark sets ⇒ lower-bound
// - Identifies necessary action sets for reaching goal
// - Disjunctive action landmarks: at least one must be used
// - Action landmark: must be in any valid plan

// *Critical Path*: $h_m$ focuses on m most difficult subgoals
// - $h_1 = h_max; h_∞$ = optimal cost in relaxed problem
// - Merges mutexes to create semi-relaxed problems

#llm[
  = Logic & FOL Resolution

  *Propositional logic*: SAT solvers (DPLL, WalkSAT)
  - *Conjunctive Normal Form (CNF)*: $(A ∨ B) ∧ (¬C ∨ D)$
  - Clause learning improves backtracking efficiency

  *Resolution*: refutation complete for CNF
  - Resolve clauses: $(A ∨ B)$ and $(¬A ∨ C)$ → $(B ∨ C)$
  - Complete for refutation (proving unsatisfiability)
  - Forward/backward chaining for Horn clauses
]

== First-Order Logic (FOL)

=== Vocabulary & Symbols
- *Constant*: Names a single object (A, B, ...) - Denotes $cal(I)(c) in cal(U)$ in model
- *Predicate*: k-ary relation (P, Q, R) - Denotes $cal(I)(P) subset cal(U)^k$
- *Function*: k-ary mapping (f, g, h) - Denotes $cal(I)(f): cal(U)^k -> cal(U)$
- *Variable*: Placeholder (x, y, z) - Assignment $sigma(x) in cal(U)$
- *Connective*: $not, and, or, =>$, $<=>$ - Boolean operations
- *Quantifier*: $forall$ ("for all"), $exists$ ("there exists") - Range: $cal(U)$

=== Syntax
*Terms* (denote objects):
- Constant or variable: A, x
- Function application: f(t_1,...,t_k)

*Atomic formula*: P(t_1,...,t_k) or t_1 = t_2

*Sentences* (closed formulas):
- Built with connectives on formulas
- Quantifiers on variables: $forall x alpha$, $exists y beta$

=== Semantics
*Model* $M = (cal(U), cal(I))$:
- *Universe* $cal(U)$: non-empty set of objects
- *Interpretation* $cal(I)$: maps symbols to denotations

*Term evaluation*:
$
  [[c]]^m_sigma = cal(I)(c), quad\
  [[x]]^m_sigma = sigma(x), quad\
  [[f(t_1,...,t_k)]]^m_sigma = cal(I)(f)([[t_1]]^m_sigma,...)\
$

*Satisfaction/Truth* (for assignment σ and model m):
$
  m,sigma models P(t_1,...,t_k) quad "iff" quad
  ([[t_1]]^m_sigma,...) in cal(I)(P)
$

$
  m,sigma models (t_1=t_2) quad "iff" quad
  [[t_1]]^m_sigma = [[t_2]]^m_sigma
$

A sentence $alpha$ is true in m $(m models alpha)$ iff $m,sigma models alpha$ for every $sigma$

=== Semantic Notions
- *Satisfiable*: $exists m: m models alpha$ -- has at least one model
- *Unsatisfiable/Contradiction*: Not satisfiable
- *Valid/Tautology*: $forall m: m models alpha$ -- write $models alpha$
- *Entailment*: $Gamma models beta$ if every model of $Gamma$ is also a model of $beta$
- *Logical equivalence*: $alpha equiv beta <=> models (alpha <=> beta)$

=== Laws & Tips
- *De Morgan*: $not forall x alpha equiv exists x not alpha$ and $not exists x alpha equiv forall x not alpha$
- *Quantifier Shift*: $forall x (alpha and beta) equiv (forall x alpha) and beta$ (when x not in β)
- *Universal Instantiation*: From $forall x alpha$ infer $alpha[t/x]$
- *Existential Generalization*: From $alpha[t/x]$ infer $exists x alpha$
- *Skolemization*: $exists x forall y P(x,y) -> forall y P(f(y),y)$

=== Applications
- Knowledge representation & rule-based AI
- Relational database queries (SQL is subset of FOL with finite domains)
- Formal verification/program logics
- Type systems
- Natural-language semantics

== Horn Clauses

*Horn Clause*: A clause (disjunction of literals) with exactly one positive literal
- Implication form: $alpha and beta and gamma => delta$
- Basis for many logic programming languages

*Datalog*: Horn clauses with no function symbols
- More efficient inference
- Decidable (guaranteed to terminate)

*Prolog*: Horn clauses with depth-first backward chaining
- Foundation of logic programming
- Adds extra features for handling negation, equality, and side-effects

== Skolemization

Transforms statements in FOL to statements in predicate logic

#pseudocode-list[
  + Rename vars to be unique
  + Convert $alpha => beta$ to $not alpha or beta$
  + Push in negations (not $forall x. alpha$ to $exists x. not alpha$)
  + Prenex normal form (quantifiers at beginning) (same order)
  + Replace all $exists$ with new function of enclosing var
  + Drop universal quantifiers
  + Convert to CNF
    + Move $not$ inward
    + Distribute $or$ over $and$
]

#llm[
  === Others

  *Clause*: Disjunction of literals (atom ∨ ¬atom)
]

= MDP & Reinforcement Learning

*Q-V Relationship*:
$
  V(s) = max_a Q(s,a)
$
- V(s) represents the value of following the optimal policy from state s
- Q(s,a) represents the value of taking action a in state s, then following the optimal policy
- Policy extraction from Q-values:
$
  pi(s) = "argmax"_a Q(s,a)
$

*Bellman Equation*:
$
  V(s)= max_a sum_{s′} T(s,a,s′)[ R(s,a,s') + γ V(s′)]
$
- Expected future discounted reward starting from state s
- Policy extraction:
$
  π(s)= "argmax"_a sum_{s′} T(s,a,s′)[R(s,a,s') + γ V(s′)]
$

*Value Iteration*:
$V_{k+1}=B[V_k]$ (contraction $gamma$)
#pseudocode-list[
  + Initialize $V(s) = 0$ for all $s in S$
  + repeat
    + $Delta = 0$
    + for each $s in S$:
      + $v = V(s)$
      + $V(s) = max_a sum_{s'} T(s,a,s')[R(s,a,s') + γ V(s')]$
      + $Delta = max(Delta, |v - V(s)|)$
  + until $Delta < epsilon$
  + return $V$
]

#llm[
  *Policy Iteration*: eval $pi$, then improve:
  $
    pi'(s)="argmax"_a Q^pi(s, a)
  $
  #pseudocode-list[
    + Initialize $π(s)$ arbitrarily for all $s in S$
    + repeat
      + Solve $V^pi(s) = sum_(s') T(s,pi(s),s')[R(s,pi(s),s') + gamma V^pi(s')]$
      + $"policy_stable" = "true"$
      + for each $s in S$:
        + $"old_action" = pi(s)$
        + $pi(s) = "argmax"_a sum_(s') T(s,a,s')[R(s,a,s') + gamma V^pi(s')]$
        + if $"old_action" != π(s)$ then $"policy_stable" = "false"$
    + until $"policy_stable"$
    + return $pi$
  ]
]
*Q-learning*:
$
  Q(s_t,a_t) <- Q(s_t,a_t)+alpha (r_t + gamma max_{a'} Q(s_{t+1},a') - Q(s_t,a_t))
$
#llm[
  - Model-free: learns directly from experience
  - Off-policy: learns about optimal policy while following exploratory policy
  - Guaranteed convergence with sufficient exploration

  *SARSA*: $Q(s,a) <- Q(s,a)+ alpha (r+ gamma Q(s',a') - Q(s,a))$
  - On-policy: learns about the policy being followed
  - Safer in dangerous environments with exploration

  *Temporal Difference (TD)(λ)*:
  - Balances immediate rewards with long-term returns
  - $λ=0$: Pure TD learning; $λ=1$: Monte Carlo learning
  - Eligibility traces provide credit assignment through time
]


= POMDP Approximations

*Most Likely State (MLS)*: Plan in most-likely state $s^* = "argmax"_s b(s)$
- *When works*: Unimodal, sharply peaked beliefs
- *When fails*: Diffuse/multimodal beliefs; information-gathering critical
- Approximates belief as point mass: $V(b) ≈ V_"MDP"(s^*)$
- Policy: $π(b) ≈ π_"MDP"(s^*)$

*Most Likely Observation (MLO)*: Assume most-likely obs $z^* = "argmax"_z P(z | b, a)$
- *When works*: Highly peaked observation models
- *When fails*: Noisy observations; diverse outcomes important
- Prunes observation branching in planning
- Next belief:
$
  b_(t+1)(s_(t+1)) & = Pr(s_(t+1) | b_t, a_t, z_t) \
                   & prop Pr(s_(t+1)|b_t, a_t) Pr(z_t|s_(t+1)) \
                   & = sum_(s_t)
                     underbrace(Pr(s_(t+1) | s_t, a_t) b_t (s_t), "Transition Update")
                     dot underbrace(Pr(o_t | s_(t+1), a_T), "Obs. Update")
$

#llm[
  *Quick-MDP Approximation (QMDP)*: Use MDP $Q's ⇒ a^*= "argmax"_a sum_s b(s)Q_"MDP"(s,a)$
  - Assumes perfect state knowledge after one step
  - Ignores observation process in planning
  - No value for information-gathering actions

  *Other Approaches*:
  - *Receding Horizon Control (RHC)*: Finite horizon H, execute first action, repeat
  - *Point-based Value Iteration (PBVI)*: Sample belief points, backup over samples
  - *POMCP*: Monte-Carlo tree search for POMDPs
]

= Motion Planning

== #link("https://youtu.be/Ob3BIJkQJEw")[RT (Random Tree) Algorithms]

- Regular RT: selects random (not nearest) node to extend

=== *Rapidly-exploring Random Tree (RRT)*
- Probabilistically complete (finds path, not necessarily optimal)
- Biases search toward unexplored regions
- Effectively handles high-dimensional spaces
- Typically leads to jagged paths since tree is never overwritten

#pseudocode-list[
  + Initial config (starting tree or node $q_"init"$)
  + Repeat until goal within range or max-iters $K$ \
    + Sample point randomly $q_"rand"$\
    + Find nearest node in tree $q_"near"$ \
    + Extend nearest node with a new node $q_"new"$ distance $Delta q$ along path to sample (if there is a collision, place the new node before the collision)
]

=== *RRT\**
- Optimal version of RRT that rewires tree for better paths
- Guaranteed asymptotic optimality

== #link("https://youtu.be/6sautP4cSkQ")[Probabilistic Roadmap Method (PRM)]

PRM is a way of BUILDING a graph from a known C-Space (and then a search algorithm can be run on top of it).
- Probabilistically complete

#pseudocode-list[
  + Sample n collision-free points
  + Connect nearby points with collision-free paths
  + Search resulting roadmap with A\* or similar
]

- *Multi-query*: Reuse roadmap for multiple planning problems
- *PRM\**: Asymptotically optimal variant with connection radius
- *Variants*: LazyPRM, EST, BIT*, FMT*

= CSP Algorithms

== CSP Definitions
$
  cal(X) & := "set of variables" \
         & = {X_1, ..., X_n} \
  cal(D) & := "sets of domains" \
         & = {D_1, ..., D_n} ("ex:" D_i = {r, g, b}) \
  cal(C) & := "set of constraints" \
         & = {C_1, ..., C_n}
$
Where $C_i$ contains a scope (tuple of vars ex: $(X_1, X_4, X_7)$) and a relation (legal values, ex: $X_1 != X_4 != X_7$)

*Arc*: Directed binary constraint between two variables, denoted $C_{X Y}$
- Represents constraint from variable X to variable Y
- Consistent if for every value of X, there exists a compatible value for Y

*Consistent Assignment*: For variables $X,Y$ and constraint $C_{X Y}$, an assignment to $X$ is consistent if:
$forall x in D(X), exists y in D(Y) : (x,y) "satisfies" C_{X Y}$

*Arc Consistency*: A directed constraint $C_{X Y}$ is arc consistent if all values in $D(X)$ have at least one consistent value in $D(Y)$

== Arc Consistency-3 (AC-3)

#pseudocode-list[
  + queue $:= {"All arcs" C_(X Y)}$
  + while queue:
    + $(X,Y) <-$ queue.pop()
    + if revise(X,Y): // Removes values from D(X) that have no support in D(Y)
      + if D(X) empty:
        + return failure
      + Add all arcs $(Z,X)$ to queue where $Z != Y$
  + return success
]

#llm[
  == Others
  *Heuristics*: MRV (min remaining values) + Degree; Min-conflicts good for N-Queens
  - *Minimum Remaining Values (MRV)*: choose variable with fewest legal values
  - *Degree*: tie-breaker, choose variable with most constraints
  - *Least Constraining Value (LCV)*: assign value that rules out fewest choices
  - *Min-conflicts*: local search, flip to minimize violations

  *Global constraints*: specialized algorithms for common patterns
  - AllDiff: efficient for distinct value constraints
  - Symmetry breaking reduces search space
  - Constraint propagation algorithms vary by constraint type
]

= *Monte Carlo Tree Search (MCTS)*
*Goal/Theory*
- Always have a "best route/sequence of actions" estimation even if stopped prematurely
- Starting from root, start expanding the tree, giving each node an estimated value $V_i$ by simulating a random path ("rolling out") from that node and updating it if a child's random rollout is determined to be better than the parent's
- Good when branching factor is high.

*Algorithm*
1. Selection -- Use max $"UCB1"(S_i) := overline(V_i) + C sqrt(ln(N)/n_i)$ from root downward to determine next node to expand

  $i := "State or node index"$ \
  $overline(V_i) := "Avg Value" = t_i / n_i$,\
  $quad t_i := "Total value (of child nodes)"$ \
  $C := "Exploration const. higher = more explore"$,\
  $N := "Nodes Visited"$,\
  $n_i := "times" i "visited"$

2. Expansion -- If $i$ visited ($n_i > 0$), and thus has an estimated value, expand to $i$'s children and evaluate

3. Simulation -- From a starting node, take random actions (or some default policy) until reaching terminal node with value.

4. Backpropogation -- Update statistics based on value or terminal node up to root (when using $"UCB1"$, this means updating the counts and total values)

// = SAT & DPLL

// *Davis–Putnam–Logemann–Loveland (DPLL)*: Choose lit, unit-prop, backtrack
// - Unit propagation forces deterministic assignments
// - Pure literal elimination optimizes search
// - Modern SAT solvers add clause learning, restarts

// = Typst Math Reminders
// - Use *named operators*: `int`, `sum`, `Pi` (not `prod`)
// - Greek letters via names (`alpha`, `beta`, …) – *no quotes needed*
// - Quote only *multi-character* variables/functions: `"bel"`, `"eta"`
// - Fractions: `(num)/(den)`; integrals: `integral ... dif x`
// - Tables: commas separate cells; matrices use semicolons
//
= PDDL

*Structure*: Domain file (rules/actions) + Problem file (instance)
- *Domain*: Types, predicates, functions, actions, constants
- *Problem*: Objects, initial state, goal state

*Components*:
- *Types*: Hierarchical categorization (e.g., `truck airplane - vehicle`)
- *Predicates*: Boolean state relations (e.g., `(at ?obj ?loc)`)
- *Actions*: State transformations with:
  - Parameters: Objects acted upon
  - Preconditions: Required state for execution
  - Effects: State changes after execution

*Syntax Example*:

```lisp
(define (domain logistics)
    :effect (and (not (at ?v ?from)) (at ?v ?to)))
    :precondition (at ?v ?from)
    :parameters (?v - vehicle ?from ?to - location)
  (:action move
  (:predicates (at ?obj ?loc))
  (:types truck airplane - vehicle)
```

*Advanced Features*:
- *Numeric Fluents*: Continuous quantities (`:functions`)
- *Durative Actions*: Actions with time (`?duration`, `:at start/end/over`)
- *Derived Predicates*: High-level properties (`:derived`)
- *Conditional Effects*: Context-dependent outcomes (`when`)

*Advantages*:
- Declarative: Specify what, not how
- Domain-problem separation enables reuse
- Formal semantics for automated analysis
- Standard interface to planning algorithms

*Admissible*: Doesn't ever give an overestimation to the goal.

#set page(paper: "us-letter", margin: 1cm, columns: 2)

#figure(caption: "Inference Algorithms")[#rotate(0deg)[#image(
  "inference-algs.png",
)]]

#figure(caption: "Search Algorithms")[#rotate(0deg)[#image("search-algs.png")]]

#figure(caption: "Inference Algorithms")[#rotate(0deg)[#image(
  "max-product.png",
)]]

// #link("Barber.pdf")[Textbook Reference]

#align(center)[— End of Cheat Sheet —]
