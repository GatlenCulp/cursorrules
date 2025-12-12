# PSet PDF to Typst

Please convert this problem set PDF into a typst code block. Keep the content verbatum. Do not solve, just write empty `#solution[ TODO ]` and `#question(status: "todo")[...]` for each. Event if there is no title in the original PDF, give a title based on the content of the question or what it is trying to show. You don't need to add "Part" labels.

Here is an example:

```typst
#import "../preamble.typ": *

#show: homework.with(
  title: "HW07",
  date: datetime(year: 2025, month: 10, day: 27),
)

#let (render, Cell, In, Out) = callisto.config(
  nb: json("hw07.ipynb"),
)

#set heading(numbering: "1.1")


// Math shorthands
#let KL(u, v) = {
  $D_"KL" (#u || #v)$
}

#let refPol = $overline(pi)$

= Actor-Critic Methods

#question(status: "done")[
  For convenience, we define the discounted total reward of a trajectory $tau$ as:
  $
    R(tau) := sum_(t=0)^oo gamma^t r(s_t, a_t),
  $ <eq:reward-def>
  where $s_t, a_t$ are the state-action pairs in $tau$. Observe that:
  $
    V_mu (pi_theta) = EE_(tau tilde P_(mu)^(pi_theta)) [R(tau)],
  $ <eq:value-def>
  For some initial state distribution, $gamma = 1$ and a finite-horizon ending at $T$, we have seen in class that
  $
    grad_theta V_mu (pi_theta) = EE_(tau tilde P_(mu)^(pi_theta)) [R(tau) sum_(t=0)^(T-1) grad_theta log pi_theta (a_t | s_t)].
  $ <eq:policy-grad-finite>
  In this problem, we will show that a similar expression can be derived for the gradient of the value function in terms of action-value functions $Q$. This introduction of the critic $Q$ lends itself to actor-critic methods, both basic and advanced. Assume that policies are stochastic.
]

#pagebreak()
== Part 1 <sec:part1>

#question(status: "done")[
  Derive the policy gradient for the discounted infinite horizon case (note that the derivation in lecture is for the finite horizon one). In other words, prove that for a discount factor $gamma < 1$, the policy gradient is
  $
    grad_theta V_mu (pi_theta) = EE_(tau tilde P_(mu)^(pi_theta)) [R(tau) sum_(t=0)^oo grad_theta log pi_theta (a_t | s_t)]
  $ <eq:policy-grad-infinite>
]

#solution[
  For simplicity, we will alias some of the variables:
  $
    pi = pi_theta, quad P = P^(pi_theta)_mu, quad grad = grad_theta, quad V_mu (pi_theta) = V (theta)
  $
  We can use the same likelihood rescaling trick from lecture.
  From above:
  $
    V(theta) = EE_(tau ~ P) [R(tau)]
  $
  Where $R(tau)$ completely handles the $gamma$ term and infinite sum. We are interested in the policy gradient:
  $
    grad_theta V(theta) = evaluated(grad_Delta V(theta + Delta))_(Delta = 0)
  $

  Firstly, we can get $V(theta + Delta)$ by using likelihood rescaling -- the discounted total trajectory is scaled according to the relative likelihood of generating that trajectory given the change in parameters:
  $
    V(theta + Delta) = EE_(tau ~ P) [R(tau) (product_(t) pi_(theta + Delta) (a_t | s_t))/(product_(t) pi_(theta) (a_t | s_t))]
  $
  To find $evaluated(grad_Delta V(theta + Delta))_(Delta = 0)$, first note that the expectation is independent of $Delta$, so it can be moved in immediately. $Delta$ is also unrelated to $R(tau)$. So:

  $
    evaluated(grad_Delta V(theta + Delta))_(Delta = 0)
    = evaluated(
      EE_(tau ~ P) [
        R(tau)/(product_(t) pi_(theta) (a_t | s_t))
        grad_Delta (product_(t) pi_(theta + Delta) (a_t | s_t))
      ]
    )_(Delta = 0)
  $

  Focusing just on term with the gradient:
  $
    grad_Delta (product_(t) pi_(theta + Delta) (a_t | s_t))
    = sum_(t=0)^oo [
      grad_Delta pi_(theta + Delta) (a_t | s_t)
      product_(t' != t) (pi_(theta + Delta) (a_t' | s_t'))
    ]
  $
  Substiting:
  $
    evaluated(grad_Delta V(theta + Delta))_(Delta = 0)
    = evaluated(
      EE_(tau ~ P) [
        R(tau)/(product_(t) pi_(theta) (a_t | s_t))
        sum_(t=0)^oo [
          grad_Delta pi_(theta + Delta) (a_t | s_t)
          product_(t' != t) (pi_(theta + Delta) (a_t' | s_t'))
        ]
      ]
    )_(Delta = 0) \
    \
    = EE_(tau ~ P) [
      R(tau)/(product_(t) pi_(theta) (a_t | s_t))
      sum_(t=0)^oo [
        ( grad_theta pi_(theta) (a_t | s_t))
        product_(t' != t) (pi_(theta) (a_t' | s_t'))
      ]
    ]
    \
    \ "Make prod independent of t and factor it out" \
    = EE_(tau ~ P) [
      R(tau)/(product_(t) pi_(theta) (a_t | s_t))
      sum_(t=0)^oo [
        (grad_theta pi_(theta) (a_t | s_t))
        (product_(t') (pi_(theta) (a_t' | s_t'))) / (pi_(theta) (a_t | s_t))
      ]
    ]
    \
    = EE_(tau ~ P) [
      R(tau)/(product_(t) pi_(theta) (a_t | s_t))
      (product_(t) pi_(theta) (a_t | s_t))
      sum_(t=0)^oo [
        (grad_theta pi_(theta) (a_t | s_t)) / (pi_(theta) (a_t | s_t))
      ]
    ]
    \
    = EE_(tau ~ P) [
      R(tau)/(product_(t) pi_(theta) (a_t | s_t))
      (product_(t) pi_(theta) (a_t | s_t))
      sum_(t=0)^oo [
        (grad_theta pi_(theta) (a_t | s_t)) / (pi_(theta) (a_t | s_t))
      ]
    ]
    \
    = EE_(tau ~ P) [
      R(tau)
      sum_(t=0)^oo [
        (grad_theta pi_(theta) (a_t | s_t)) / (pi_(theta) (a_t | s_t))
      ]
    ]
  $

  Since $grad_theta log pi_theta = (grad_theta pi_theta) / pi_theta$:

  $
    grad_theta V_mu (pi_theta) = EE_(tau ~ P) [
      R(tau)
      sum_(t=0)^oo
      grad_theta log pi_theta (a_t | s_t)
    ]
  $

  $qed$

  Note: This didn't actually feel easier than the common proof, I'm not sure how the jump from Likelihood scaling to chain rule worked.
]

#pagebreak()
== Part 2 <sec:part2>

#question(status: "done")[
  Derive the policy gradient for the actor-critic for the discounted infinite horizon case, that is, show that policy gradients can be expressed as
  $
    grad_theta V_mu (pi_theta) = EE_(tau tilde P_(mu)^(pi_theta)) [sum_(t=0)^oo gamma^t Q^(pi_theta) (s_t, a_t) grad_theta log pi_theta (a_t | s_t)]
  $ <eq:actor-critic-1>
  and
  $
    grad_theta V_mu (pi_theta) = 1/(1-gamma) EE_(s tilde d_(mu)^(pi_theta)) EE_(a tilde pi_theta (dot | s)) [Q^(pi_theta) (s, a) grad_theta log pi_theta (a | s)],
  $ <eq:actor-critic-2>
  where the discounted future state distribution $d_(mu)^(pi_theta)$ is defined as
  $
    d_(mu)^(pi_theta) (s) = (1-gamma) sum_(t=0)^oo gamma^t PP(s_t = s | pi_theta, mu)
  $ <eq:state-dist>

  _Note: $d$ should have an overline_
]

#solution[
  For simplicity, we will alias some of the variables:
  $
    pi = pi_theta, quad P = P^(pi_theta)_mu, quad grad = grad_theta, quad V_mu (pi_theta) = V (theta), quad Q^(pi_theta) = Q, quad d^(pi_theta)_mu = d
  $

  *Proving @eq:actor-critic-1*

  Let's start from the derivation in part 1 and expand the product of sums:

  $
    grad_theta V_mu (pi_theta)
    = EE_(tau ~ P) [
      R(tau)
      sum_(t=0)^oo
      grad_theta log pi_theta (a_t | s_t)
    ]
    \
    = EE_(tau ~ P) [
      (sum_(t=0)^oo gamma^t r(s_t, a_t))
      (sum_(t'=0)^oo grad_theta log pi_theta (a_(t') | s_(t')))
    ]
    \
    = EE_(tau ~ P) [
      sum_(t=0)^oo sum_(t'=0)^oo
      gamma^t r(s_t, a_t)
      grad_theta log pi_theta (a_(t') | s_(t'))
    ]
    \
    = EE_(tau ~ P) [
      sum_(t'=0)^oo
      grad_theta log pi_theta (a_(t') | s_(t'))
      sum_(t=0)^oo
      gamma^t r(s_t, a_t)
    ]
    \
    = EE_(tau ~ P) [
      sum_(t'=0)^oo
      grad_theta log pi_theta (a_(t') | s_(t'))
      (
        sum_(t=0)^(t'-1)
        gamma^t r(s_t, a_t)
        + sum_(t=t')^(oo)
        gamma^t r(s_t, a_t)
      )
    ]
  $

  The first term represents rewards before time $t'$, which are independent of action $a_(t')$. By the causality principle, this term contributes zero to the gradient. Therefore:

  $
    = EE_(tau ~ P) [
      sum_(t'=0)^oo
      grad_theta log pi_theta (a_(t') | s_(t'))
      sum_(t=t')^(oo)
      gamma^t r(s_t, a_t)
    ]
  $

  Reindex the inner sum with $k = t - t'$, so $t = t' + k$:

  $
    = EE_(tau ~ P) [
      sum_(t'=0)^oo
      grad_theta log pi_theta (a_(t') | s_(t'))
      sum_(k=0)^(oo)
      gamma^(t'+k) r(s_(t'+k), a_(t'+k))
    ]
    \
    = EE_(tau ~ P) [
      sum_(t'=0)^oo
      gamma^(t')
      grad_theta log pi_theta (a_(t') | s_(t'))
      sum_(k=0)^(oo)
      gamma^k r(s_(t'+k), a_(t'+k))
    ]
  $

  Note that the inner sum is the Q-function:
  $
    Q^pi (s_(t'), a_(t')) = EE[sum_(k=0)^(oo) gamma^k r(s_(t'+k), a_(t'+k)) | s_(t'), a_(t')]
  $

  Therefore (renaming $t'$ back to $t$):
  $
    grad V (pi) = EE_(tau ~ P) [
      sum_(t=0)^oo
      gamma^t Q(s_t, a_t)
      grad log pi(a_t | s_t)
    ]
  $

  This proves @eq:actor-critic-1. $qed$

  #colbreak()
  *Proving @eq:actor-critic-2*

  Where the discounted future state is:
  $
    d(s) = (1-gamma) sum_(t=0)^oo gamma^t PP(s_t = s | pi, mu)
  $


  We will work backward from @eq:actor-critic-2 to get @eq:actor-critic-1 which we have shown above to be an equivalent formulation of the policy gradient.
  $
    grad V (pi) = 1/(1-gamma) EE_(s ~ d)
    EE_(a ~ pi(dot | s)) [
      Q (s, a) grad log pi (a | s)
    ]
    \
    = 1/(1-gamma) [
      integral_s (1-gamma) sum_(t=0)^oo gamma^t PP(s_t = s | pi, mu)
      EE_(a ~ pi(dot | s)) [
        Q (s, a) grad log pi (a | s)
      ]
      dd(s)
    ]
    \
    =
    integral_s sum_(t=0)^oo gamma^t PP(s_t = s | pi, mu)
    ( integral_a pi(a | s)
      Q (s, a) grad log pi (a | s)
      dd(a))
    dd(s)
    \
    =
    integral_s integral_a
    sum_(t=0)^oo gamma^t
    PP(s_t = s | pi, mu) pi(a | s)
    Q (s, a) grad log pi (a | s)
    dd(a)
    dd(s)
    \
    =
    integral_s integral_a
    sum_(t=0)^oo gamma^t
    PP(s_t = s, a_t = a | pi, mu)
    Q (s, a) grad log pi (a | s)
    dd(a)
    dd(s)
    \
    =
    sum_(t=0)^oo gamma^t
    integral_s integral_a
    PP(s_t = s, a_t = a | pi, mu)
    Q (s, a) grad log pi (a | s)
    dd(a)
    dd(s)
    // PLEASE EXPLAIN THIS JUMP
    \
    =
    sum_(t=0)^oo gamma^t
    integral_tau
    PP(tau | pi, mu)
    Q (s_t, a_t) grad log pi (a_t | s_t)
    dd(tau)
    \
    =
    sum_(t=0)^oo gamma^t
    integral_tau
    PP(tau | pi, mu)
    Q (s_t, a_t) grad log pi (a_t | s_t)
    dd(tau)
    \
    =
    sum_(t=0)^oo gamma^t
    EE_(tau ~ P)[
      Q (s_t, a_t) grad log pi (a_t | s_t)
    ]
    \
    =
    EE_(tau ~ P)[
      sum_(t=0)^oo gamma^t
      Q (s_t, a_t) grad log pi (a_t | s_t)
    ]
  $

  $qed$
]

#pagebreak()
= Relationship between soft Q-learning and policy gradient <sec:soft-q>

#question(status: "partial")[
  Entropy-regularized reinforcement learning is widely used in reality, especially in recent advances of LLM post-training. Instead of the traditional discounted return $sum_(t=0)^oo gamma^t r_t$, we consider the following entropy-regularized return function: given a reference policy $refPol$, the total return is defined as
  $
    sum_(t=0)^oo gamma^t {
      r(s_t, a_t)
      - tau KL(pi(dot | s_t), refPol(dot | s_t))
    },
  $ <eq:entropy-return>
  where $KL(p, overline(p)) := sum_a p(a) log(p(a) slash overline(p)(a))$ denotes the KL-divergence between two distributions, and characterizes the difference between these two distributions. Hence maximizing the entropy-regularized return @eq:entropy-return not only aims for maximizing the total return of policy $pi$, but also keeps the policy $pi$ close to the reference policy $refPol$.   We define the value function of policy $pi$ at state $s$ to be the expected entropy-regularized return:
  $
    V_pi (s) := EE [sum_(t=0)^oo {gamma^t r(s_t, a_t) - tau KL(pi(dot | s_t), refPol(dot | s_t))} | s_0 = s],
  $ <eq:entropy-value>
  and similarly the Q-function of policy $pi$:
  $
    Q_pi (s, a) := EE [
      r(s_0, a_0)
      + sum_(t=1)^oo gamma^t
      {
        r(s_t, a_t)
        - tau KL(pi(dot | s_t), refPol(dot | s_t))
      }
      | s_0 = s, a_0 = a
    ].
  $ <eq:entropy-q>
  This gives the relationship
  $
    V_pi (s) = EE_(a tilde pi(dot | s)) [
      Q_pi (s, a)
    ]
    - tau KL(pi(dot | s), refPol(dot | s)).
  $ <eq:v-q-relation>
][
  Note: $tau$ is the temperature parameter, not trajectory. $tau = 0 =>$ greedy. $tau -> oo =>$ reference policy. Used to weight the KL-divergence cost.
]

#pagebreak()
== Part 1: Boltzman policy <sec:boltzman>

#question(status: "done")[
  Given a Q-function, show that the optimizer of the optimization problem
  $
    arg max_pi {
      EE_(a tilde pi(dot | s)) [Q_pi (s, a)]
      - tau KL(pi(dot | s), refPol(dot | s))
    }
  $ <eq:boltzman-opt>
  over all policies satisfies
  $
    pi_Q^cal(B) (a | s) = refPol(a | s) dot (exp(Q(s, a) slash tau))/(EE_(a tilde refPol(dot | s)) [exp(Q(s, a) slash tau)]), quad forall a in cal(A).
  $ <eq:boltzman-policy>
  The policy $pi_Q^cal(B)$ is called the Boltzman policy.
]

#solution[
  Expand the optimized expression
  $
    EE_(a tilde pi(dot | s)) [Q_pi (s, a)]
    = sum_a pi(a | s) Q_pi (s, a) \
    KL(pi(dot | s), refPol(dot | s))
    = sum_a pi(a | s) log(pi(a | s)/(refPol(a | s)))
  $
  Combining:
  $
    EE_(a tilde pi(dot | s)) [Q_pi (s, a)]
    - tau KL(pi(dot | s), refPol(dot | s)) \
    = sum_a pi(a | s) Q_pi (s, a)
    - tau sum_a pi(a | s) log(pi(a | s)/(refPol(a | s)))
    \
    = sum_a pi(a | s) (Q_pi (s, a) - tau log(pi(a | s) / (refPol(a | s))))
  $

  We will assume a fixed $Q$, independent from $pi$. Maximizing this quantity is a constrained optimization problem below:
  $
        max quad & sum_a pi(a | s) (
                     Q_pi (s, a) - tau log(pi(a | s) / (refPol(a | s)))
                   ) \
    "w.r.t" quad & sum_a pi(a | s) = 1
  $
  $
    cal(L)
    = sum_a pi(a | s) (
      Q (s, a) - tau log(pi(a | s) / (refPol(a | s)))
    )
    + lambda (1 - sum_a pi(a | s))
    \
    = sum_a pi(a | s) (
      Q (s, a) - tau log pi(a | s) + tau log refPol(a | s)
    )
    + lambda (1 - sum_a pi(a | s))
    \
    "Abusing notation by dropping most arguments for simplicity:"
    \
    = sum_a pi (
      Q - tau log pi + tau log refPol
    )
    + lambda (1 - sum_a pi(a | s))
    \
  $
  Lets take the derivative of the policy at some fixed action $a$ and state $s$. Only one element from each of the sums will not be a constant:
  $
    \
    pdv(cal(L), pi(a | s))
    = 0 = [
      pi (tau/pi)
      + 1 (Q - tau log pi + tau log refPol)
    ]
    + lambda (- 1)
    \
    = [
      tau + Q - tau log pi + tau log refPol
    ]
    - lambda
    \
    = Q + tau (1 - log pi + log refPol) - lambda
  $
  Solving for $pi$:
  $
    lambda - Q = tau (1 - log pi + log refPol) \
    => (lambda - Q)/tau - 1 - log refPol = - log pi \
    => log pi = (Q - lambda)/tau + 1 + log refPol \
    => pi = e^((Q - lambda)/tau + 1 + log refPol) \
    = refPol exp((Q - lambda + tau)/tau) \
    = refPol exp(Q / tau) exp((tau - lambda) / tau) \
    = refPol exp(Q / tau) exp((tau - lambda) / tau) \
  $

  #sub-solution[

    We can now introduce the probability constraint:
    $
      1 = sum_a pi(a | s) = sum_a refPol(a | s) exp(Q(s, a) / tau) exp((tau - lambda) / tau) \
    $
    And we can solve for $exp((tau - lambda) / tau)$, the constant:
    $
      1 = exp((tau - lambda) / tau) sum_a refPol(a | s) exp(Q(s, a) / tau) \
      => exp((tau - lambda) / tau) = 1 / (sum_a refPol(a | s) exp(Q(s, a) / tau)) \
    $

  ]
  Plugging this term in:
  $
    pi = refPol exp(Q / tau) exp((tau - lambda) / tau) \
    = refPol exp(Q / tau) / (sum_a refPol(a | s) exp(Q(s, a) / tau))
  $

  This provides us with the Boltzman policy:
  $
    pi_Q^cal(B) (a | s) = refPol(a | s) dot (exp(Q(s, a) slash tau))/(EE_(a tilde refPol(dot | s)) [exp(Q(s, a) slash tau)]), quad forall a in cal(A).
  $
]

#pagebreak()
== Part 2: Value function representation <sec:value-rep>

#question(status: "done")[
  Given a Q-function, show that the value function of the Boltzman policy has the following representation:
  $
    V_(pi_Q^B) (s) = tau log EE_(a tilde refPol(dot | s)) [exp(Q(s, a) / tau)].
  $ <eq:boltzman-value>
]

#solution[
  We simply need to plug in the Boltzman policy $pi = pi_Q^cal(B)$ into equation for $V_pi$ and letting $Q_pi$ instead be some arbitrary $Q$:
  $
    V_pi (s) = EE_(a tilde pi(dot | s)) [
      Q(s, a)
    ]
    - tau KL(pi(dot | s), refPol(dot | s)).
  $
  And letting the constant denominator be $C := EE_(a tilde refPol(dot | s)) [exp(Q(s, a) slash tau)]$
  $
    pi_Q^cal(B) (a | s)
    = refPol(a | s) dot
    (exp(Q(s, a) slash tau))
    / (EE_(a tilde refPol(dot | s)) [exp(Q(s, a) slash tau)])
    , quad forall a in cal(A) \
    = 1/C refPol(a | s) exp(Q(s, a) slash tau)
    , quad forall a in cal(A).
  $

  Starting with the *first term*:
  $
    EE_(a tilde pi(dot | s)) [
      Q(s, a)
    ]
    = sum_a pi(a | s) dot Q(s, a)
    \
    = sum_a
    refPol(a | s) dot
    (exp(Q(s, a) slash tau))
    / (EE_(a tilde refPol(dot | s)) [exp(Q(s, a) slash tau)])
    dot Q(s, a)
  $
  For simplicity lets call this constant denominator $C$
  $
    = sum_a
    refPol(a | s) dot
    (exp(Q(s, a) slash tau)) / C
    dot Q(s, a)
    \
    = 1/C sum_a
    refPol(a | s) Q(s, a) exp(Q(s, a) / tau)
    \
  $

  For the *second term*, KL-divergence:
  $
    KL(pi(dot | s), refPol(dot | s))
    = sum_a pi(a | s) log (pi(a | s))/(refPol(a | s)) \
    = sum_a
    1/C refPol(a | s) exp(Q(s, a) slash tau)
    [
      log (1/C refPol(a | s) exp(Q(s, a) slash tau))
      - log refPol (a | s)
    ]
    \
    = 1/C sum_a
    refPol(a | s) exp(Q(s, a) slash tau)
    [
      log 1/C + log refPol(a | s) + Q(s, a) slash tau
      - log refPol (a | s)
    ]
    \
    = 1/C sum_a
    refPol(a | s) exp(Q(s, a) slash tau)
    [
      log 1/C + Q(s, a) slash tau
    ]
    \ \
    = log(1 slash C)/C sum_a
    refPol(a | s) exp(Q(s, a) / tau) \
    + 1/(C tau) sum_a
    refPol(a | s) exp(Q(s, a) / tau) Q(s, a)
    \
    \ \
    = log(1 slash C)
    + 1/(C tau) sum_a
    refPol(a | s) exp(Q(s, a) / tau) Q(s, a)
    \
  $
  Combining the two terms:
  $
    V_pi (s)
    = [
      1/C sum_a
      refPol(a | s) Q(s, a) exp(Q(s, a) / tau)
    ] \
    - tau [
      log(1 slash C)
      + 1/(C tau) sum_a
      refPol(a | s) exp(Q(s, a) / tau) Q(s, a)
    ]
    \ \
    = [
      (1/C - tau 1/(C tau)) sum_a
      refPol(a | s) Q(s, a) exp(Q(s, a) / tau)
    ]
    - tau log(1 slash C)
    \ \
    = - tau log(1/C)
    \
    = tau log C
    \
    = tau log EE_(a tilde refPol(dot | s)) [exp(Q(s, a) / tau)]
  $
  This gives us the expected:
  $
    V_(pi_Q^B) (s) = tau log EE_(a tilde refPol(dot | s)) [exp(Q(s, a) / tau)].
  $
  $qed$
]

#pagebreak()
== Part 3: Single-Step Soft Q-learning <sec:soft-q-single>

// Corrections: https://piazza.com/class/mey9hjom2td4p3/post/222#

// Use semigradient, i.e. when calculating the gradient, treat y_t as a constant



#question(status: "todo")[
  In the following, we parametrize the policy Q-function $Q_theta$ by parameter $theta$, and we denote the Boltzman policy $pi_(Q_theta)^B$ by $pi_theta$, and its value function $V_(pi_(Q_theta)^B)$ as $V_theta$. We let
  $
    y_t = r_t + gamma V_(pi_Q^B) (s_(t+1)).
  $ <eq:target>
  Show that the gradient of the soft Q-learning objective at a single sample can be written as
  $
    grad_theta [1/2 norm(Q_theta (s_t, a_t) - y_t)^2]
    \
    = (tau grad_theta log pi_theta (a_t | s_t) + grad_theta V_theta (s_t)) \
    dot (tau dot log (pi_theta (a_t | s_t))/(refPol(a_t | s_t)) - tau KL(pi_theta (dot | s_t), refPol(dot | s_t)) - delta_t),
  $ <eq:soft-q-grad>
  where $delta_t$ is defined as
  $
    delta_t = y_t - (tau KL(pi_theta (dot | s_t), refPol(dot | s_t)) + V_theta (s_t)).
  $ <eq:delta>
]

// I got the same thing but with a term of
//  from the
//  term. Did I do something wrong or is it okay to assume that
//  even thought it depends on
// ?
// 0
// Zeyu Jia  2 days ago
// You can just let \nabla_\theta y_t = 0. When calculating the gradient of \theta, just treat y_t as a constant

#solution[
  Some simplification first:
  $
    grad [1/2 norm(Q(s_t, a_t) - y_t)^2]
    = 1/2 grad (Q(s_t, a_t) - y_t)^2 \
    = (Q(s_t, a_t) - y_t) grad (Q(s_t, a_t) - y_t) \
    = (Q(s_t, a_t) - y_t) (grad Q(s_t, a_t) - underbrace(grad y_t, 0 "according to piazza")) \
    = (Q(s_t, a_t) - y_t) grad Q(s_t, a_t) \
  $
  Expanding: $y_t$
  $
    y_t = r_t + gamma V_(pi_Q^B) (s_(t+1)) \
    = r_t + gamma tau log EE_(a tilde refPol(dot | s)) [exp(Q(s, a) / tau)] \
    = r_t + gamma tau log sum_a refPol(a | s) exp(Q(s, a) / tau)
  $

  We want to eliminate $grad Q$ and replace it with $grad V$. We can do this through the definition given above:
  $
    V_theta (s) = EE_(a tilde pi_theta (dot | s)) [
      Q_theta (s, a)
    ]
    - tau KL(pi_theta (dot | s), refPol(dot | s))
    \
    grad V_theta (s) = grad EE_(a tilde pi_theta (dot | s)) [
      Q_theta (s, a)
    ]
    - grad tau KL(pi_theta (dot | s), refPol(dot | s))
    \
    =
    grad sum_a pi_theta (a | s) Q_theta (s, a)
    - tau grad KL(pi_theta (dot | s), refPol(dot | s))
    \
  $
  Focusing on the *first term*:
  $
    grad sum_a pi_theta (a | s) Q_theta (s, a)
    = sum_a [
      pi_theta (a | s) grad Q_theta (s, a)
      + Q_theta (a | s) grad pi_theta (s, a)
    ]
    \
    = sum_a
    pi_theta (a | s) [grad Q_theta (s, a)
      + Q_theta (a | s) grad log pi_theta (s, a)
    ]
  $

  And for the *second term*:
  $
    grad KL(pi_theta (dot | s), refPol(dot | s))
    = sum_a
    grad (
      pi_theta (a | s)
      log (pi_theta (a | s))/(refPol(a | s))
    )
    \
    = sum_a
    (
      pi_theta (a | s)
      grad log (pi_theta (a | s))/(refPol(a | s))
      + log (pi_theta (a | s))/(refPol(a | s)) grad pi_theta (a | s)
    )
    \
    = sum_a
    (
      pi_theta (a | s)
      [grad log (pi_theta (a | s)) - grad log (refPol(a | s))]
      + log (pi_theta (a | s))/(refPol(a | s)) grad pi_theta (a | s)
    )
    \
    = sum_a
    (
      pi_theta (a | s)
      [grad log pi_theta (a | s)]
      + log (pi_theta (a | s))/(refPol(a | s)) grad pi_theta (a | s)
    )
    \
    = sum_a
    (
      pi_theta (a | s)
      [grad log pi_theta (a | s)]
      + log (pi_theta (a | s))/(refPol(a | s)) pi_theta (a | s) grad log pi_theta (a | s)
    )
    \
    = sum_a
    pi_theta (a | s)(
      grad log pi_theta (a | s)
      + log (pi_theta (a | s))/(refPol(a | s)) grad log pi_theta (a | s)
    )
    \
    = sum_a
    pi_theta (a | s) grad log pi_theta (a | s) (
      1
      + log (pi_theta (a | s))/(refPol(a | s))
    )
  $

  *Combining*
  $
    grad V_theta (s)
    =
    grad sum_a pi_theta (a | s) Q_theta (s, a)
    - tau grad KL(pi_theta (dot | s), refPol(dot | s))
    \ \
    = sum_a
    pi_theta (a | s) [grad Q_theta (s, a)
      + Q_theta (a | s) grad log pi_theta (s, a)
    ]
    \
    - tau [
      sum_a
      pi_theta (a | s) grad log pi_theta (a | s) (
        1
        + log (pi_theta (a | s))/(refPol(a | s))
      )
    ]
    \ \
    = sum_a pi_theta (a | s)
    [\
      grad Q_theta (s, a)
      + Q_theta (a | s) grad log pi_theta (s, a)
      - tau (
        grad log pi_theta (a | s) (
          1
          + log (pi_theta (a | s))/(refPol(a | s))
        )
      )
      \
    ]
  $


  *Final Steps*

  1. And then we can solve for $grad Q_theta$ to only have an expression involving $grad V_theta, grad log pi_theta, "and" Q_theta$.
  2. Substitute $Q_theta$ from the Boltzmann policy definition
  3. Pull $delta_t$ into a separate term



  // Solving for $grad Q_theta$:
  // $
  //   = sum_a pi_theta (a | s) grad Q_theta
  //   + sum_a pi_theta (a | s)
  //   [\
  //     Q_theta (a | s) grad log pi_theta (s, a)
  //     - tau (
  //       grad log pi_theta (a | s) (
  //         1
  //         + log (pi_theta (a | s))/(refPol(a | s))
  //       )
  //     )
  //     \
  //   ]
  //   \ \
  //   =>
  //   sum_a pi_theta (a | s) grad Q_theta = grad V_theta (s)
  //   - sum_a pi_theta (a | s)
  //   [\
  //     Q_theta (a | s) grad log pi_theta (s, a)
  //     - tau (
  //       grad log pi_theta (a | s) (
  //         1
  //         + log (pi_theta (a | s))/(refPol(a | s))
  //       )
  //     )
  //     \
  //   ]
  // $
]

#pagebreak()
== Part 4: Relationship between soft Q-learning and policy gradient <sec:soft-q-pg>

// Is the delta_t inside or outside the gradient in the first term on the right hand side? I believe it is inside

#question(status: "todo")[
  If we assume that the samples $s_t, a_t, r_t$ are all collected according to policy $pi$, show the following relationship between the gradient of soft Q-learning and the policy gradient.
  // https://piazza.com/class/mey9hjom2td4p3/post/235 first term of expectation should have grad_theta log not just grad_theta
  $
    underbrace(grad_theta EE_pi [1/2 norm(Q_theta (s_t, a_t) - y_t)^2] |_(pi = pi_theta), "Q-learning gradient") \
    = underbrace(
      tau dot EE_pi [-grad_theta pi_theta (a_t | s_t) delta_t + tau grad_theta KL(pi(dot | s_t), refPol(dot | s_t))] |_(pi = pi_theta), "policy gradient"
    ) \
    + underbrace(1/2 grad_theta EE_pi [norm(V_theta (s_t) - hat(V)_t)^2] |_(pi = pi_theta), "value function gradient"),
  $ <eq:soft-q-pg-relation>
  where $hat(V)_t$ is defined as
  $
    hat(V)_t = y_t - tau KL(pi(dot | s_t), refPol(dot | s_t)) |_(pi = pi_theta).
  $ <eq:v-hat>
]

#solution[
  *To solve*
  1. Start with result from part three above
  2. Expand terms
  3. Use $delta_t = hat(V)_t - V_theta (s_t)$
  4. Use definition of the norm and expectation to put into required form.
]

#pagebreak()
= REINFORCE with cartpole <sec:cartpole>

#question(status: "done")[
  In this problem, we will try solving the cart-pole problem via a policy space method using neural networks. The setting is as follows. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

  - *Action space.* ${0, 1}$, where $0$ means the cart is pushed to the left, and $1$ means the cart is pushed to the right.

  - *State space.* (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity). In particular,
    - The cart position can take values between $(-4.8, 4.8)$, but the episode terminates if the cart leaves the $(-2.4, 2.4)$ range.
    - The pole angle can be observed between $(-0.418, 0.418)$ radians (or $plus.minus 24 degree$), but the episode terminates if the pole angle is not in the range $(-0.2095, 0.2095)$ (or $plus.minus 12 degree$). The starting pole angle is sampled uniformly from $(-0.05, 0.05)$.

  - *Rewards.* Our goal is to keep the pole upright for as long as possible. So a reward of "$+1$" for every step taken, including the termination step, is allotted.

  - *Episode Termination.* The episode terminates if any one of the following occurs:
    - Pole Angle is greater than $plus.minus 12 degree$.
    - Cart Position is greater than $plus.minus 2.4$ (center of the cart reaches the edge of the display).
    - Episode length is greater than $500$.

  Please read the detailed instructions in the Colab link (can be found here). Complete the code, do the experiments, and answer the questions in the notebook. For submission, please include all your results in the pdf submitted to Gradescope, as well as a link with your used code.
]
// To confirm my understanding of task (d), should I train the linear model using REINFORCE with temporal structure and baseline, and also train the non-linear model using REINFORCE with temporal structure and baseline, and then compare them based on the number of parameters and their policy performance (i.e., their learning curves)? hw7 Yes, you can just train the new model on a linear policy and compare it to the results from part (b).


// What is the average rewards we should get in 4-a/b. I am not sure if my algorithm is correct. Can we get a baseline?
// I got 100-300 for vanilla (probably due to randomization in the training) and 500 consistently for temporal and baseline+temporal.


#solution[
  #link(
    "https://colab.research.google.com/drive/1aYiajNFYrMbDbco8kGTqnoQjtjdDAjro?usp=sharing",
  )[Link to code: https://colab.research.google.com/drive/1aYiajNFYrMbDbco8kGTqnoQjtjdDAjro?usp=sharing
  ]

  In case there are complications, I used Callisto to display the notebook in the PDF below.
]

#pagebreak()
#set heading(numbering: none)
#set text(size: 10pt)
#render()
```
