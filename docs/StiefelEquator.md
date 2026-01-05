<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"> </script>

The concentration of measure on Stiefel manifolds has a natural geometric interpretation that generalizes the sphere case.

## The Equator for $V_{n,2}$

Given a 2-frame $X_0 = [v_1, v_2] \in V_{n,2}$, the **equator** is:

$$
\mathcal{E}(X_0) = {Y \in V_{n,2} : X_0^T Y = 0}
$$

where the condition $X_0^T Y = 0$ (a $2 \times 2$ zero matrix) means:

$$
v_i^T w_j = 0 \quad \text{for all } i,j \in {1,2}
$$

if $Y = [w_1, w_2]$.

### Geometric Interpretation

The equator consists of **all 2-frames whose span is orthogonal to the span of $X_0$**:

$$
\text{span}(Y) \perp \text{span}(X_0)
$$

Since $\text{span}(X_0)$ is a 2-dimensional subspace of $\mathbb{R}^n$, its orthogonal complement is $(n-2)$-dimensional. Therefore:

$$
\mathcal{E}(X_0) \cong V_{n-2,2}
$$

The equator is itself a Stiefel manifold - the space of 2-frames living entirely in the orthogonal complement.

## Dimensional Analysis

- $\dim(V_{n,2}) = 2n - 3$
- $\dim(\mathcal{E}(X_0)) = \dim(V_{n-2,2}) = 2(n-2) - 3 = 2n - 7$
- **Codimension** of equator: $4 = 2^2$

Compare this to the sphere:

- $\dim(S^n) = n$
- $\dim(\text{equator}) = n - 1$
- **Codimension**: $1 = 1^2$

## Concentration for Large $n$

For large $n$, measure concentrates on $\mathcal{E}(X_0)$ by the following probabilistic argument:

If $Y$ is a uniformly random 2-frame, each entry of $X_0^T Y$ is approximately Gaussian with mean 0 and variance $1/n$. Therefore:

$$
|X_0^T Y|_F = O(1/\sqrt{n}) \to 0
$$

as $n \to \infty$. Almost all frames $Y$ satisfy $X_0^T Y \approx 0$, meaning they lie approximately on the equator.

## General Pattern for $V_{n,k}$

The pattern extends naturally:

**For $X_0 \in V_{n,k}$, the equator is:**

$$
\mathcal{E}(X_0) = {Y \in V_{n,k} : X_0^T Y = 0} \cong V_{n-k,k}
$$

- **Codimension**: $k^2$
- **Interpretation**: Frames in the orthogonal complement of $\text{span}(X_0)$
- **Concentration**: For large $n$, almost all measure lies within $O(k/\sqrt{n})$ of the equator

## Connection to Geodesic Distance

The geodesic distance on $V_{n,k}$ relates to the singular values of $X_0^T Y$. The "maximally distant" frames (those on the equator) satisfy:

$$
\text{all singular values of } X_0^T Y \text{ equal zero}
$$

In the Grassmannian picture (projecting to $k$-planes via $\pi: V_{n,k} \to G_{n,k}$), the equator projects to the set of $k$-planes orthogonal to $\text{span}(X_0)$, which is isomorphic to $G_{n-k,k}$.

## Intuition

Just as on the sphere, where most vectors are nearly perpendicular to any fixed vector in high dimensions, on $V_{n,2}$ most 2-frames have their entire 2-dimensional span nearly orthogonal to any fixed 2-frame's span. The "typical" configuration is maximal independence - frames living in complementary subspaces.

This is the manifestation of the **curse of dimensionality** in geometric form.
