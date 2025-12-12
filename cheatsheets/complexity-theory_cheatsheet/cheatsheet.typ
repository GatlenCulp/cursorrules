
// #import "@preview/algorithmic:0.1.0" DOESN'T WORK FOR MY VERSION.
// #import algorithmic: algorithm

// #import "@preview/algo:0.3.6": algo, i, d, comment, code
#import "@preview/lovelace:0.3.0": pseudocode-list
#set page(paper: "us-letter", margin: 1cm, columns: 3)
#set text(size: 6pt)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  set align(center)
  set text(size: 8pt)
  it
}
#show heading.where(level: 2): it => {
  set align(center)
  set text(size: 7pt)
  it
}

#show heading.where(level: 3): it => {
  set align(center)
  set text(size: 6pt, fill: blue)
  it
}

#let st = $"s.t."$

// Math Control
#let If = $#strong("if")$
#let Then = $#strong("then")$
#let Else = $#strong("else")$


// ---- Complexity Theory Specifics -----
// #algo(
//   title: "Fib",
//   parameters: ("n",),
// )[
//   if $n < 0$:#i\ // use #i to indent the following lines
//   return null#d\ // use #d to dedent the following lines
//   if $n = 0$ or $n = 1$:#i #comment[you can also]\
//   return $n$#d #comment[add comments!]\
//   return #smallcaps("Fib")$(n-1) +$ #smallcaps("Fib")$(n-2)$
// ]

#let enc(x) = $angle.l #x angle.r$

// Langs
#let Lang(x) = $sans(#x)$
#let lang = $Lang(L)$

// Turing Machines
#let Tm(x) = $mono(#x)$

#let tm = $Tm(M)$
#let TM = $Tm("TM")$
#let TMs = $Tm("TMs")$

#let NTM = $Tm("NTM")$
#let NTMs = $Tm("NTMs")$

#let utm = $Tm(U)$
#let UTM = $Tm("UTM")$
#let UTMs = $Tm("UTMs")$

// Language Classes
// #let langClass(x) = $#upper($#x$)$
#let langClass(x) = $#x$
#let TIME = $langClass("TIME")$
#let NTIME = $langClass("NTIME")$

#let NP = $langClass("NP")$
#let NPHARD = $langClass("NP-HARD")$
#let NPCOMPLETE = $langClass("NP-COMPLETE")$

#let coNP = $langClass("coNP")$
#let coNPHARD = $langClass("coNP-HARD")$
#let coNPCOMPLETE = $langClass("coNP-COMPLETE")$

#let BPP = $langClass("BPP")$
#let PSPACE = $langClass("PSPACE")$

#let EXPTIME = $langClass("EXPTIME")$

#let DECIDABLE = $#langClass("DECIDABLE")$
#let UNDECIDABLE = $#langClass("UNDECIDABLE")$
#let RECOGNIZABLE = $#langClass("RECOGNIZABLE")$
#let UNRECOGNIZABLE = $#langClass("UNRECOGNIZABLE")$


// Operations
#let pReducesTo = $scripts(<=)_p$


// Accept/Reject
#let True = text(strong("True"), fill: green)
#let False = text(strong("False"), fill: red)
#let accept = text(strong("accept"), fill: green)
#let accepts = text(strong("accepts"), fill: green)
#let reject = text(strong("reject"), fill: red)
#let rejects = text(strong("rejects"), fill: red)

// Misc
#let poly = $"poly"$

#heading(
  level: 1,
  numbering: none,
)[6.1400 Cheat Sheet \ (Computability & Complexity)]
#align(center)[
  Gatlen Culp (gculp\@mit.edu)
  #sym.dot 2025-05-21

  #rect[
    = Legend
    $
         Tm("Monospace") & = "Turing Machines" \
      Lang("Sans-serif") & = "Languages" \
       langClass("CAPS") & = "Language Classes" \
    $
  ]
]

= Regular Languages

Closure under:
$
  A union B, quad A sect B, quad A^R, quad A^*, quad not A, \
  A dot B, quad A - B = A sect (not B)
$

Order of Operations: $* -> dot -> +$

= DFA

$
  "DFA" = (Q, Sigma, delta, q_0, F)
$

== DFA $->$ MinDFA

=== Defining MinDFA and Ideas

_Note: The minimum DFA is unique_

A *MinDFA* requires (1) no indistinguishable states, (2) no unreachable states

$
          tm_p & := tm "but starting on state" P \
               \
  w in Sigma^* & "distinguishes states" p, q \
               & := tm_p "and" tm_q "have different outputs on" w \
               \
         p ~ q & := "states" p, q "are distinguishable" \
               & := exists w "distinguishing" p, q \
$

Distinguishable states form a partitions of equivalence on $Q$. These can be simplified after removing unreachable states.

=== Table-Filling Algorithm (DFA $->$ MinDFA)

#figure()[#image("mindfa.jpg")]

// TODO: Find better one.

= NFA

$
  "NFA" = (Q, Sigma, delta, Q_0, F)
$

== NFA $->$ DFA

Create a table with n + 1 columns -- one for the DFA state and one for each of $n$ transitions. List out the transitions in each column. Add rows for each new possible set of states reachable.

#figure()[#image("nfatofdfa.png")]

- Note: No bijection between ${1, ..., n} "and" underbrace(2^{1, ..., n}, "powerset")$

== NFA $->$ GNFA $->$ RegExp

#pseudocode-list[
  + *if* $|"states"| = 2$:
    + return $R(q_"start", q_"acc")$
  + $G' <- G."copy"()$
  + pick $q_"rip" in (Q - {q_"start", q_"acc"})$
  + $Q' <- Q \\ q_"rip"$
  + $forall q_i, q_j in (Q' \\ q_"acc", Q' \\ q_"start")$:
    + $
        R'(q_i, q_j) <- &[R(q_i, q_"rip") dot underbrace(R(q_"rip", q_"rip")^*, "Self-loops") dot R(q_"rip", q_j)] \
        &+ R(q_i, q_j)
      $
    + Repeat from the top
]

= Streaming Algorithms

#pseudocode-list[
  + Initialize vars and their assignments on vars
  + When next symbol $= sigma$:
    + Run pseudocode for $sigma$
  + Accept/reject based on vars
]

= Myhill-Nerode Theorem

$
  forall lang in "Langs": "either" \
  exists "DFA recognizing" lang \
  "or" \
  "There are infinite strings to"\ "trick DFA attempting to recognize" lang
$

ie: $"equiv classes of" equiv_L "is finite" <=> lang "is regular"$

== Distinguishing Set

$
  "Strings" w_1, ..., w_n, underbrace(..., "possibly"\ "infinite") st forall i != j:\
  exists z st (w_i z in lang) "xor" (w_j z in lang)
$

*Ex*: For $lang = {0^n 1^n | n >= 0}$\
take $S = {0, 00, ... 0^n, ...}$ where $z = 1^i$

Then $forall w_i, w_j in S: quad [(0^i 1^i) in lang] and [(0^j 0^i) in.not lang]$

== Streaming Distinguisher

*Streaming Distinguisher $D_n$*: Distinguishing set when limiting the concatenated strings to length $n$.

$
  (forall "distinct" x, y in D_n)(exists "word" z) \
  (x z in lang) "xor" (y z in lang)\
  and (|x z|, |y z| <= n)
$

If $forall n, exists D_n st |D_n| >= 2^(S(n))$\
then all streaming algs must also use $>= s(n)$

= Communication Complexity

== Protocol
$
  A, B: Sigma^* times (Sigma^* union "STOP")
$
Which collectively compute $f$

$
  "If" exists "streaming alg using" <= S(m) "space on" 2m "inputs then:"\
  "cc"(f) <= O(s(m))
$

== Switcheroo Lemma

$
  (x, y) "and" (x', y') "share comms pattern" P, \
  "then so do" (x, y') "and" (x', y)
$

=== Example: ALICE-HAS-MORE Lower Bound

#text(fill: blue)[
  For $"ALICE-HAS-MORE"(A, B)$, create set $S = ((1^(i+1)0^(n-i-1), 1^i 0^(n-i)) : 0 <= i <= n-1)$.

  All pairs in $S$ have $"ALICE-HAS-MORE" = 1$.

  If protocol uses $< log_2 n$ bits, by pigeonhole, two pairs share same pattern:
  $(1^(i+1)0^(n-i-1), 1^i 0^(n-i))$ and $(1^(j+1)0^(n-j-1), 1^j 0^(n-j))$.

  By switcheroo: $(1^(i+1)0^(n-i-1), 1^j 0^(n-j))$ and $(1^(j+1)0^(n-j-1), 1^i 0^(n-i))$ also share same pattern.

  This implies $i+1 > j$ and $j+1 > i$, contradiction when $i != j$.
]

= Turing Machines

$
  (tm in TMs) = (Q, Sigma, Gamma, delta, q_o, q_"acc", q_"rej")
$

(1) Read $sigma$ (2) Write $sigma$ (3) Move left/right (4) Change states

= Equivalence Classes
$
  (x script(equiv)_lang y) &:= (forall z in Sigma^*) [x z in lang <=> y z in lang] \
  &<=> x, y "are" lang "equivalent" \
  &<=> x, y "are distinguishable to" lang
$

This is similar to equivalence relation of DFA states. "Further transitions are identical."

_Note: Equivalence classes require (1) Symmetric, (2) Transitive, (3) Reflexive_

== Under TMs
$
  Delta_tm & : Sigma^x -> Q \
           & := "State you end up in after reading a string"
$

$
  x approx_tm y & <=> Delta(x) = Delta(y) \
                & <=> x equiv_lang y \
                & <=> x "and" y "are distinguishable strings under" tm
$

Where I'm pretty sure $tm$ is the machine recognizing $lang$

= Formal Systems & Their Limits

*Formal system* $F$
- finite language, notion of *proof*, notion of *truth*

*"Interesting"* iff
1. *Expressive* – each TM–input pair $(M, w)$ maps (computably) to a sentence $S_(M,w)$ with
  $S_(M,w)$ *true* $arrow.r.double$ $M$ accepts $w$.
2. *Checkable* – the set $\{(S, P) | P$ _is a proof of_ $S$ *in* $F\}$ is *decidable*.
3. *Halting-decisive* – if $M$ halts on $w$, then $F$ proves either $S_(M,w)$ or $not S_(M,w)$.

*Consistency* – no statement $S$ with both $S$ and $not S$ provable.
*Completeness* – every statement $S$ has either $S$ or $not S$ provable.

*Limit theorems*
- *Gödel (1931)*: any consistent, interesting $F$ is *incomplete* (true but unprovable statements exist).
- *Gödel (1931)*: $F$ cannot prove its own consistency.
- *Church–Turing (1936)*: deciding whether a sentence of $F$ has a proof is *undecidable*.

= Recursion Theorem
// (Lec 12)

WLOG, a program can reference its its own code
$
  forall t: Sigma^* times Sigma^* -> Sigma^*\ "(Write" t "as if it already has recursion)"\
  \
  exists R: Sigma^* -> Sigma^* "s.t." forall R(w) = t(enc(R), w)\
$

= Turing Machine Minimization
// (Lec 13?)
($UNDECIDABLE$)
$
  "Where" Tm(M) in TMs:\
  "Minimal"(enc(M)) <=> M "is a minimal state TM"
$

= Fixed Point Theorem
// (Lec 13?)
If $t: Sigma^* -> Sigma^*$ is computable:\
$quad$ $exists Tm(F) in TMs$ s.t. \
$quad$ $t(enc(Tm(F)))$ outputs $enc(Tm(G))$ where $lang(Tm(F)) = lang(Tm(G))$

ie: We can make a copy that is behaviorally identical to the input for any string-to-string function $t$.

= Rice's Theorem
// (Lec 14?)
"Program analysis is hard"
$
  "let" P & := "Property" \
          & : TMs -> {0 ,1} \
    and P & "is Non-Trivial (Not all acc/rej)" \
    and P & "is Semantic" (lang(Tm(M)) = lang(Tm(M')) => P(Tm(M)) = P(Tm(M'))) \
   "then" & {enc(M) | P(Tm(M)) = 1} in UNDECIDABLE
$

= Decidable Predicates
// (Lec 15)

$
  R in "Predicates", quad Tm(M) in TMs \
  R(x, y) "is" True <=> Tm(M)(x, y) accepts
$
Vaguely:\
$NTMs$ = $RECOGNIZABLE$\
$TMs$ = $DECIDABLE$.

= Computational Complexity
$
         T(n) & := "Time complexity of" tm in TMs \
              & "on strings up to length" n \
   TIME(t(n)) & := {lang | lang "has time complexity" O(t(n))} \
  NTIME(t(n)) & := {lang | lang "is decided by" O(t(n)) "time complexity" NTM} \
            P & := "Poly Time" = union_(k in NN) TIME(n^k) \
           NP & := "Non-Deterministic Poly Time" = union_(k in NN) NTIME(n^k)
$

= Universal Turing Machines
$
  (utm in UTMs) & = Tm("Universal TM") \
            utm & : "Code" times "Input" -> "Output" \
            utm & "on" enc("tm, x"), accept "if" (tm accepts x) "else" reject
$
(A) Input tape, (B) State Tape, (C) Code Tape, (D) Simulation Tape

= Time Hierarchy Theorem
"You can solve strictly more problems /w more time"
$
  If g(n) > n^2 f(n)^2\
  Then TIME(f(n)) subset TIME(g(n))
$


= Extended Church Turing Thesis
All intuitions about efficient algs = Poly-time $TMs$

= Mapping Reduction
$
  A <=_m B := (exists f: Sigma^* -> E^*) \
  st w in A <=> f(w) in B
$

== Poly-Time Reduction $pReducesTo$

Same as mapping reduce, but in poly-time (uses partial order)

= $NPHARD$ and $NPCOMPLETE$

$
      NPHARD & := {lang | forall Lang(A) in NP, Lang(A) pReducesTo lang} \
  NPCOMPLETE & := {lang | lang in NP and \
             & lang in NPHARD "ie: not" EXPTIME "or smthn"}
$

= Non-Deterministic TMs ($NTMs$)

- May proceed according to several possible transitions
- Accepts if there is any accepting condition

= Verifier Characterization of $NP$

$
  lang in NP\
  <=>\
  exists k in RR, V(x, y) in {"Poly-time predicates"} st\
  L = {x | exists y in Sigma^*[ |y| <= k|x|^k and V(x, y) ]}
$
ie: Guess and check with some poly-time verifier

something something
$
  V "ran in" poly(n) "time" <=> lang in TIME(2^O(k^n) dot poly(n))
$

= Oracles
// TODO: Fix
$
  langClass(X)^Lang(Y) & := "Problems" \
                       & "solvable using any TM in" X "with an" \
                       & "using an oracle for the language" Lang(Y) \
                       & "(or selecting any oracle from" langClass(Y))
$

= BPP & Error-Reduction
$
  BPP := { L | L "is decided by a"\ "poly-time probabilistic TM with error" <= 1 / 3 }
$
Any constant $< 1/2$ works: once the error is bounded away from $1/2$ by $1 / poly(n)$ we can *amplify*.

*Error-reduction lemma*
Let $epsilon in (0, 1/2)$ and $k in NN$.
If machine $M_1$ runs in $t(n)$ and

$"Pr"(M_1(x) != L(x)) <= 1 / 2 - epsilon,$

there exists $M_2$ with

$
  "Pr"(M_2(x) != L(x)) & < 1 / 2^(n^k), \
           "time"(M_2) & = O(n^k dot t(n) / epsilon^2).
$

Construction: run $M_1$ independently $m := Theta(n^k / epsilon^2)$ times and output the majority vote.

*Chernoff bound* (additive form)
For independent $0/1$ $X_i$, let $X = sum_i X_i$, $mu = E[X]$:

$
  "Pr"(X < (1 - delta) mu) <= exp(-delta^2 mu / 2).
$

With $X_i = 1$ iff $M_1$ is correct $=> mu >= (1/2 + epsilon) m$.
Choose $delta = 2 epsilon / (1 + 2 epsilon)$ to obtain

$
  "Pr"("majority wrong") & <= exp(-2 epsilon^2 m) \
                         & < 2^(- epsilon^2 m / ln 2).
$

Setting $m = 10 n^k / epsilon^2$ yields error $< 2^(-n^k)$.

#figure()[#image("np-complete.png")]

// #pagebreak()
// #set page(paper: "us-letter", margin: 1cm, columns: 2)
#set text(size: 5pt)
= Problems
#let SAT = $langClass("SAT")$
#let SAT3 = $langClass("3SAT")$
#let HALT = $langClass("HALT")$
#let NHALT = $langClass("NHALT")$
#let CLIQUE = $langClass("CLIQUE")$
#let IS = $langClass("IS")$
#let VERTEXCOVER = $langClass("VERTEX-COVER")$
#let SUBSETSUM = $langClass("SUBSET-SUM")$
#let KNAPSACK = $langClass("KNAPSACK")$
#let HOM = $langClass("HOM")$
#let HOMEDGE = $langClass("HOM-EDGE")$
#let PARTITION = $langClass("PARTITION")$
#let BINPACKING = $langClass("BIN-PACKING")$
#let SHORTESTPATH = $langClass("SHORTEST-PATH")$
#let HAMPATH = $langClass("HAM-PATH")$
#let HAMCYCLE = $langClass("HAM-CYCLE")$
#let TAUT = $langClass("TAUT")$
#let TAUTOLOGY = $langClass("TAUTOLOGY")$
#let UNSAT = $langClass("UNSAT")$
#let FACTORING = $langClass("FACTORING")$
#let FIRSTSAT = $langClass("FIRST-SAT")$
#let MINFORMULA = $langClass("MIN-FORMULA")$
#let PRIMES = $langClass("PRIMES")$
#let CIRCUITSAT = $langClass("CIRCUIT-SAT")$
#let LOWWEIGHT2SAT = $langClass("LOW-WEIGHT-2SAT")$
#let MAXCLIQUE = $langClass("MAX-CLIQUE")$
#let MINWEIGHT2SAT = $langClass("MIN-WEIGHT-2SAT")$
#let MINCNF = $langClass("MIN-CNF")$
#let TQBF = $langClass("TQBF")$
#let ADFA = $langClass("A_{DFA}")$
#let ANFA = $langClass("A_{NFA}")$
#let EQDFA = $langClass("EQ_{DFA}")$
#let EQNFA = $langClass("EQ_{NFA}")$
#let ATM = $langClass("A_{TM}")$
#let EQTM = $langClass("EQ_{TM}")$
#let EMPTYTM = $langClass("EMPTY_{TM}")$
#let HALTTM = $langClass("HALT_{TM}")$
#let NOHAMPATH = $langClass("NO-HAMPATH")$
#let EQUIV = $langClass("EQUIV")$
#let NEQUIV = $langClass("NEQUIV")$
#let PM1LI = $langClass("+-1-LI")$
#let LONGPATH = $langClass("LONGEST-PATH")$
#let GG = $langClass("GG")$
#let FG = $langClass("FG")$
#let PATH = $langClass("PATH")$

#let boolForm = math.phi.alt
#let classin = $space.en "in"$

$
  PRIMES &:= {n | n "is prime"} \
  &"in" P\
  PATH &:= {(G, s, t) | "Graph" G "has a path from" s "to" t} \
  &"in" P\
  SAT &:= {boolForm | boolForm "is a satisfiable boolean formula"} \
  &"in" NPCOMPLETE\
  SAT3 &:= {boolForm | boolForm "is a satisfiable 3-CNF formula"} \
  &"in" NPCOMPLETE\
  CIRCUITSAT &:= {C | "Boolean circuit" C "is satisfiable"} \
  &"in" NPHARD\
  HALT &:= {enc("Tm(M), w") | Tm(M) "halts on input" w} \
  &"in" RECOGNIZABLE "but" UNDECIDABLE\
  NHALT &:= {enc("Tm(M), w") | Tm(M) "does not halt on input" w} \
  &"in" UNRECOGNIZABLE\
  CLIQUE &:= {(G, k) | "Graph" G "has a clique of size" >= k} \
  &"in" NPCOMPLETE\
  MAXCLIQUE &:= {(G, k) | "Max clique size in graph" G "is exactly" k} \
  &"in" P^NP\
  IS &:= {(G, k) | "Graph" G "has an independent set of size" >= k} \
  &"in" NPCOMPLETE\
  VERTEXCOVER &:= {(G, k) | "Graph" G "has a vertex cover of size" <= k} \
  &"in" NPCOMPLETE\
  SUBSETSUM &:= {(S, t) | exists S' subset S "s.t." sum_(x in S') x = t} \
  &"in" NPCOMPLETE\
  KNAPSACK &:= {(I, v, w, W, V) | "Item set" I "with values" v "and weights" w} \
  &"has subset with weight" <= W "and value" >= V \
  &"in" NPCOMPLETE\
  HOM &:= {(G, H) | "Exists homomorphism from graph" G "to graph" H} \
  &"in" NPCOMPLETE\
  HOMEDGE &:= {(G, H) | "Exists edge-preserving homomorphism from" G "to" H} \
  &"in" P\
  PARTITION &:= {S | exists S' subset S "s.t." sum_(x in S') x = sum_(x in S - S') x} \
  &"in" NPCOMPLETE\
  BINPACKING &:= {(S, k, B) | "Items" S "can fit in" k "bins of capacity" B} \
  &"in" NPCOMPLETE\
  SHORTESTPATH &:= {(G, s, t, k) | "Graph" G "has path from" s "to" t "of length" <= k} \
  &"in" P\
  LONGPATH &:= {(G, s, t, k) | "Graph" G "has path from" s "to" t "of length" >= k} \
  &"in" NPCOMPLETE\
  HAMPATH &:= {(G, s, t) | "Graph" G "has Hamiltonian path from" s "to" t} \
  &"in" NPCOMPLETE\
  NOHAMPATH &:= {(G, s, t) | "Graph" G "has no Hamiltonian path from" s "to" t} \
  &"in" "coNP-COMPLETE"\
  HAMCYCLE &:= {G | "Graph" G "has a Hamiltonian cycle"} \
  &"in" NPCOMPLETE\
  LOWWEIGHT2SAT &:= {(boolForm, k) | "2-CNF formula" boolForm "has satisfying assignment with <= k true variables"} \
  &"in" NPCOMPLETE\
  MINWEIGHT2SAT &:= {(boolForm, k) | "Min weight of satisfying assignment for 2-CNF formula" boolForm "is" k} \
  &"in" P^NP\
  PM1LI &:= {(A, b) | "System" A x = b "has a solution with entries in" {-1, 0, 1}} \
  &"in" NPCOMPLETE\
  TAUT = TAUTOLOGY &:= {boolForm | boolForm "is a tautology"} \
  &"in" "coNP-COMPLETE"\
  UNSAT &:= {boolForm | boolForm "is unsatisfiable"} \
  &"in" "coNP-COMPLETE"\
  FACTORING &:= {(n, k) | n "has a non-trivial factor" <= k} \
  &"in" NP sect coNP\
  FIRSTSAT &:= {boolForm | "First assignment in lex order satisfies" boolForm} \
  &"in" "P^NP-COMPLETE"\
  MINFORMULA &:= {(boolForm, k) | boolForm "has equivalent formula of size" <= k} \
  &"in" "coNP^NP-COMPLETE"\
  MINCNF &:= {(boolForm, k) | boolForm "has equivalent CNF formula of size" <= k} \
  &"in" "coNP^NP-COMPLETE"\
  TQBF &:= {boolForm | "Quantified boolean formula" boolForm "is true"} \
  &"in" "PSPACE-COMPLETE"\
  GG &:= {"Generalized Geography game"} \
  &"in" "PSPACE-COMPLETE"\
  FG &:= {"Formula Game"} \
  &"in" "PSPACE-COMPLETE"\
  NEQUIV &:= {(M_1, M_2) | Lang(M_1) != Lang(M_2)} \
  &"in" NPCOMPLETE\
  EQUIV &:= {(M_1, M_2) | Lang(M_1) = Lang(M_2)} \
  &"in" "coNP-COMPLETE"\
  ADFA &:= {enc(M) | M "is a DFA that accepts some string"} \
  &"in" DECIDABLE\
  ANFA &:= {enc(M) | M "is an NFA that accepts some string"} \
  &"in" DECIDABLE\
  EQDFA &:= {enc("M_1, M_2") | M_1, M_2 "are DFAs and" Lang(M_1) = Lang(M_2)} \
  &"in" DECIDABLE\
  EQNFA &:= {enc("M_1, M_2") | M_1, M_2 "are NFAs and" Lang(M_1) = Lang(M_2)} \
  &"in" DECIDABLE\
  ATM &:= {enc(M) | M "is a TM that accepts some string"} \
  &"in" RECOGNIZABLE "but" UNDECIDABLE\
  HALTTM &:= {enc(M) | M "is a TM that halts on some input"} \
  &"in" RECOGNIZABLE\
  EMPTYTM &:= {enc(M) | M "is a TM and" Lang(M) = {}} \
  &"in" UNRECOGNIZABLE\
  EQTM &:= {enc("M_1, M_2") | M_1, M_2 "are TMs and" Lang(M_1) = Lang(M_2)} \
  &"in" UNRECOGNIZABLE\
$

= Complexity Classes
#import "@preview/cetz:0.3.2"

// TODO: Add $coNP^NP$ and $NP^coNP$

#cetz.canvas({
  import cetz.draw: *

  let trans = 97%
  let thickness = 1.2pt
  // Classes
  let computabilityRect = rect.with(
    stroke: (paint: purple.lighten(10%), thickness: thickness, dash: "dashed"),
    fill: purple.transparentize(trans),
    radius: 1pt,
  )

  let complexityRect = rect.with(
    stroke: (paint: green.lighten(10%), thickness: thickness, dash: "dashed"),
    fill: green.transparentize(trans),
    radius: 1pt,
  )

  let oracleRect = rect.with(
    stroke: (paint: blue.lighten(10%), thickness: thickness, dash: "dashed"),
    fill: blue.transparentize(trans),
    radius: 1pt,
  )

  let probRect = rect.with(
    stroke: (paint: red.lighten(10%), thickness: thickness, dash: "dashed"),
    fill: red.transparentize(trans),
    radius: 1pt,
  )

  let spaceRect = rect.with(
    stroke: (paint: orange.lighten(10%), thickness: thickness, dash: "dashed"),
    fill: orange.transparentize(trans),
    radius: 1pt,
  )

  let computability(a, b, name, ..args, body) = {
    computabilityRect(a, b, name: name, ..args)
    content(
      (name: name, anchor: "north"),
      anchor: "north",
      padding: 0.5em,
      name: name + "-label",
    )[$#body$]
  }

  let complexity(a, b, name, label: "north", ..args, body) = {
    complexityRect(a, b, name: name, ..args)
    content(
      (name: name, anchor: label),
      anchor: label,
      padding: 0.5em,
      name: name + "-label",
    )[$#body$]
  }

  let oracle(a, b, name, label: "north", ..args, body) = {
    oracleRect(a, b, name: name, ..args)
    content(
      (name: name, anchor: label),
      anchor: label,
      padding: 0.5em,
      name: name + "-label",
    )[$#body$]
  }

  let prob(a, b, name, ..args, body) = {
    probRect(a, b, name: name, ..args)
    content(
      (name: name, anchor: "north"),
      anchor: "north",
      padding: 0.5em,
      name: name + "-label",
    )[$#body$]
  }

  let space(a, b, name, ..args, body) = {
    probRect(a, b, name: name, ..args)
    content(
      (name: name, anchor: "north"),
      anchor: "north",
      padding: 0.5em,
      name: name + "-label",
    )[$#body$]
  }

  let padAmt = 0.3

  // Scale up the diagram
  scale(1.2)

  // Helper Grid
  // grid((-5, -5), (5, 5), step: 5)
  // grid((-5, -5), (5, 5), help-lines: true)

  // Classes, from center outward.
  // Basic Classes
  complexity((0.1, 0.1), (rel: (0.3, 0.3)), "Regular")[$"Regular"$]
  complexity((0, 0), (rel: (0.8, 0.8)), "P")[$P$]
  complexity(
    (rel: (-padAmt, -padAmt), to: "P.south-west"),
    (rel: (3 * padAmt, padAmt), to: "P.north-east"),
    label: "east",
    "NP",
  )[$NP$]
  complexity(
    (rel: (-padAmt, -padAmt), to: "P.south-west"),
    (rel: (padAmt, padAmt + 1.5 * padAmt), to: "P.north-east"),
    "coNP",
  )[$coNP$]
  complexity(
    (rel: (-padAmt, padAmt), to: "coNP.north-west"),
    (rel: (padAmt, -padAmt), to: "NP.south-east"),
    "BPP",
  )[$BPP$]
  // Probability
  prob(
    (rel: (-padAmt, 1.5 * padAmt), to: "BPP.north-west"),
    (rel: (padAmt, -padAmt), to: "BPP.south-east"),
    "P^NP",
  )[$P^NP$]
  // Oracles
  oracle(
    (rel: (-padAmt, padAmt), to: "P^NP.north-west"),
    (rel: (padAmt + 2 * padAmt, -padAmt), to: "P^NP.south-east"),
    label: "east",
    "NP^NP",
  )[$NP^NP$]
  oracle(
    (rel: (-padAmt, padAmt + 1.5 * padAmt), to: "P^NP.north-west"),
    (rel: (padAmt, -padAmt), to: "P^NP.south-east"),
    "coNP^coNP",
  )[$coNP^coNP$]
  // Space
  space(
    (rel: (-padAmt, padAmt), to: "coNP^coNP.north-west"),
    (rel: (padAmt, -padAmt), to: "NP^NP.south-east"),
    "PSPACE",
  )[$PSPACE$]
  // EXPTIME
  complexity(
    (rel: (0, padAmt), to: "PSPACE.north-west"),
    (rel: (0, 0), to: "PSPACE.south-east"),
    "EXPTIME",
  )[$EXPTIME$]
  // Computability
  computability(
    (rel: (0, padAmt), to: "EXPTIME.north-west"),
    (rel: (0, 0), to: "EXPTIME.south-east"),
    "DECIDABLE",
  )[$DECIDABLE$]
  computability(
    (rel: (0, padAmt), to: "DECIDABLE.north-west"),
    (rel: (0, 0), to: "DECIDABLE.south-east"),
    "RECOGNIZABLE",
  )[$RECOGNIZABLE$]

  // coNP intersection NP
  content((rel: (0, 0.5em), to: "P.north"))[$-> FACTORING$]
  // NP
  content((rel: (-1em, 0.5em), to: "NP.east"))[$-> SAT3$]
  // coNP
  content((rel: (0, -1em), to: "coNP.north"))[$-> TAUTOLOGY$]
  // P^NP
  content((rel: (0, -1em), to: "P^NP.north"))[$-> FIRSTSAT$]

  let potentialEquality = bezier.with(stroke: black.transparentize(60%))

  potentialEquality(
    "P-label.south",
    "NP-label.south",
    (rel: (0, -2em), to: "NP-label"),
  )

  potentialEquality(
    "NP-label.south",
    "coNP-label.south",
    (rel: (0, -2em), to: "NP-label"),
    stroke: black.transparentize(60%),
  )

  potentialEquality(
    "NP^NP-label.south",
    "NP-label.south",
    (rel: (0, -2em), to: "NP-label"),
    stroke: black.transparentize(60%),
  )
})


#figure()[#image("turing-reduce-empty.png")]
#figure()[#image("turing-reduce-empty-2.png")]
#figure()[#image("kolmogorov.png")]

= Complexity/Computability Relationships

Given five languages with these properties:
$
  A & : "in" P \
  B & : "in" NP \
  C & : "is NP-complete" \
  D & : "is decidable" \
  E & : "is recognizable but not decidable"
$

Determining whether the following reductions are ALWAYS, MAYBE, or NEVER true:

$
      E <=_T D & : #text(fill: red)[NEVER true] \
      B <=_p C & : #text(fill: red)[ALWAYS true] \
      A <=_p B & : #text(fill: red)[MAY BE true] \
  B <=_p not C & : #text(fill: red)[MAY BE true] \
      D <=_m C & : #text(fill: red)[ALWAYS true]
$

// Note: These hold regardless of open questions like P vs NP
