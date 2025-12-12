---
description: Audit mathematical notation in a document and add inline comments identifying issues
globs: ['*.md', '*.typ']
---

<!-- TODO: Update, this is a dumb command. -->

# Audit Mathematical Notation

Review mathematical notation and add inline comments identifying issues **unless already noted** (e.g., in a footnote).

## Categories

**(A) Abuse of Notation** - Mathematical notation being used incorrectly

- Using ( f(x) ) to denote both a function and its value
- Treating a random variable and its realization as the same
- Using ( \\sum\_{i=1}^n ) when ( n ) is not defined as an integer

**(B) Redundant Information** - Duplicate or unnecessary expressions

- Defining the same variable multiple times
- Tautological statements

**(C) Notation Collisions** - Symbol reuse for different purposes

- ( y ) for production inputs and consumer allocation
- ( n ) for sample size and iteration count

## Comment Format

**Markdown:** `<!-- CATEGORY: Explanation | Recommended: Fix -->`

**Typst:** `// CATEGORY: Explanation | Recommended: Fix`

## Guidelines

- Insert comments immediately before/after problematic notation
- Do NOT modify existing markers like `(?)`
- Recommend widely-recognized notation
- Only flag issues within current document

## Examples

### Markdown

```markdown
Let \( y \) represent production. Later, \( y \) is consumer allocation.

<!-- COLLISION: y used for both production and allocation | Recommended: Use x for allocation -->
```

```markdown
The expected value satisfies \( E[X] = E[X] \).

<!-- REDUNDANT: Tautology provides no information | Recommended: Remove or state actual property -->
```

### Typst

```typst
// ABUSE: Using = for definition | Recommended: V(s) := max_a Q(s,a)
Let $V(s) = max_a Q(s,a)$ where $V(s)$ is the value function.

// ABUSE: Use :in for membership definition, Pr needs square brackets | Recommended: s :in cal(S) and Pr[a|s]
Define $pi(a|s) = Pr(a|s)$ for all states $s in cal(S)$.
```

```typst
$
  theta_(t+1) = theta_t - alpha grad_theta L(theta_t)
$

// ABUSE: Using = for assignment | Recommended: theta_(t+1) <- theta_t - alpha grad_theta L(theta_t)
```

## Output

After auditing, provide a summary with counts: `Found X issues: A abuse, B redundant, C collisions`
