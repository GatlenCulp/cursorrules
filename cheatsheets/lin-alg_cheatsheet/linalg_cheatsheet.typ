#import "@preview/lovelace:0.3.0": pseudocode-list
#import "@preview/fletcher:0.5.4" as fletcher: diagram, edge, node
#set page(paper: "us-letter", margin: 1cm, columns: 3)
#set text(size: 10pt)
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


#heading(level: 1, numbering: none)[18.06 Cheat Sheet \ (Linear Algebra)]
#align(center)[
  Gatlen Culp (gculp\@mit.edu)
  #sym.dot May 2025
]

They don't actually have a cheatsheet, this is just for my review

// #import fletcher.shapes: diamond

// #diagram(
//   node-stroke: 1pt,
//   node((0, 0), [Start], corner-radius: 2pt, extrude: (0, 3)),
//   edge("-|>"),
//   node(
//     (0, 1),
//     align(center)[
//       Hey, wait,\ this flowchart\ is a trap!
//     ],
//     shape: diamond,
//   ),
//   edge("d,r,u,l", "-|>", [Yes], label-pos: 0.1),
// )


= Inverses

Gauss Jordan elimination for $ mat(A | I) -> mat(I | A^(-1)) $

To solve system of linear equations you can use $ mat(A | b) -> mat(I | c) $ where $c$ is the solution IF $A$ is invertible.


Let $A in FF^(n times n)$ (must be square)

$
  "Invertible(A)" & <=> "rank"(A) = n \
                  & <=> C(A) = FF^(n) \
                  & <=> C(A^T) = FF^n \
$

= Linear Systems
- RREF can be broken down into sub matrix transformations

// ### Lecture 6 (Fri Feb 14 2025)
// * In this class we covered the general solution for a system of linear equations Ax=b.
// * The basic principle: if $b$ is not in $C(A)$, then there is no solution. If $b$ is in $C(A)$, then there must exist at least one "particular solution," call it $x_0$. Then the set of all solutions to $Ax=b$ is the set of vectors $x_0 + x'$, where $x_0$ is the particular solution, and $x'$ is any vector from the null space $N(A)$.
// * General recipe for solving:
//     * given $(A|b)$, apply Gauss-Jordan elimination to transform to RREF system $(R|c)$.
//     * If the RREF system contains a row that says 0 = nonzero, then we have a contradiction, and in this case $b$ is not in $C(A)$ and there is no solution.
//     * Otherwise, set the free variables to zero to find a particular solution $x_0$.
//     * Separately, solve for the null space $N(A)$.
//     * Then the set of all solutions to $Ax=b$ is the set of vectors $x_0 + x'$, where $x_0$ is the particular solution, and $x'$ is any vector from $N(A)$.

// **Reading:** Strang 3.3.

// Skipping notes for Lecture 7-9

= Subspaces

Requirements:
- Todo

= Null Space $N(A)$

$
  N(A) := {x | forall x st A x = 0}
$

Let $A in FF^(m times n)$
- $C(A) in "Subspaces"(CC^m)$
- $N(A) in "Subspaces"(CC^n)$

Rank nullity theorem / ???:

$
  dim C(A) + dim N(A) = n
$

= Properties & Common Symbols

- *Square*
  - *Invertible*? (which of these are?)
  - *Triangular*
    - *Upper Triangular* $R$
    - *Lower Triangular* $L$
      - *Diagonal* $D$
        - *Diagonal Eigenvalue Matrix* $Lambda$ (Diag(eigenvalues))
          - For the eigenvalues of $A$, $A V = V Lambda$
  - *(Positive/Negative) Definite* $A > 0$ -- Square matrix with positive eigenvalues
    - Alternatively: x^T A x > 0
    - $"Invertible"(A) => A^T A$ is positive definite
    - Can be factored with a full rank matrix $R$ as $R^T R$
    - *(Positive/Negative) Semi-Definite* $A >= 0$ -- Same, but can have some zero eigenvalues

- *Normal*: $A A^* = A^* A$
  - *Symmetric* $S$: $S^T = S$
    - *Hermitian* $H$: $H^* = H$
      - Diagonal entries are real
  - *Orthogonal* $Q$: $Q^T = Q^(-1)$
    - *Unitary* $U$: $U^* = U^(-1)$
      - ie: $U^* U = I$
      - Eigenvalues sit on unit circle

*Eigen*
- *Eigenvector Matrix* $E$ = Eigenvectors (as columns)????

*Misc*
- *Projection* $P$
  - *Indempotent*: $P^2 = P$

*Frequency*
- *Unitary Inverse DFT* $F$
  - $F^* = overline(F) = F^(-1)$ (Unitary)
  - Typically $x$ might be some time series data about a signal (ex: $x_t$ is the reading at time $t$). So $F x$ would transform $x$ to be written in the fourier (frequency) basis.
- *Forward DFT* $K = sqrt(n) overline(F)$
- *Circulant* $C$ (right-shift in this class, $C^T$ is left shift)
  - Can be defined as a sum of a vector and shift-permutation matrix $C = "circ"(c) = sum_(k=0)^(n-1) c_k P^k$
  - Can be further broken down into $C = sum_(k=0)^(n-1) c_k P^k = F^* (sum_(k=0)^(n-1) c_k Lambda^k) F = F D F^*$ (Type of spectral decomposition)
    - Since $F P F^* = "diag"(1, omega, omega^2, ..., omega^(n-1)) = Lambda$
    - Here $D$ are diag(eigenvalues($C$)) while $Lambda$ are diag(eigenvalues($P$)). The eigenvalues of $C$ are exactly $lambda_k = (K c)_k$

= Projection Matrix

= Decompositions

- *Eigen/Spectral Decomposition* $A = Q Lambda Q^(-1)$
  - $A in FF^(n times n)$
  - $Lambda$ is capital $lambda$ hence "Eigenvalues" on the diagonal
  - $Q$ are the eigenvectors (one per row) (not necessarily normalized)
  - Type of Schur form (don't need to know what this is)
  - *Diagonalizable* -- $A$ has $n$ independent Eigenvectors (ie: You can find $E^(-1) A E = D$ where $D$ is diagonal.)
  - *Spectral Theorem* $A = E Lambda E^*$
    - Spectral decomposition but $Q$ is unitary
    - Only possible if $A$ is Hermitian

- *Singular Value Decomposition (SVD)*


- *QR Decomposition* $A = Q R$
  - $Q$ = Orthonormal
  - $R$ = Upper Triangular
  - Achieved with *Gram Schmidt*

= Least Squares

Minimizing $|| A x - b ||^2$ is $x = (A^T A)^(-1) A^T b$


= Pseudoinverse

$A^+ = V Sigma^+ U^*$

= Determinants
$det(A B) = det(A) det(B)$

= Misc
- If $A$ is full-rank that just means there are no ZERO eigenvalues.


Godly video on the fundamental spaces: https://youtu.be/ZdlraR_7cMA

// #image("https://www.cs.utexas.edu/~flame/laff/alaff-beta/images/Chapter04/FundamentalSpaces.png")
= Quadratic Form

Linear Quadratic (LQ) form is $v^T A v in FF$
- Conceptually, describes how much $v$ relates with $A$. Or as distance where $A$ dictates the geometric relationship, e.g. if $A = I$, then $v^T I v = v^T v = sum v_i^2$
- Derivative:
$
  gradient_x (x^T A x) = (A + A^T) x
$

= Vector Calculus
$
  partial / (partial x) [u A x] = A^T u
$

// Section 2 here is useful: #link("https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf")
