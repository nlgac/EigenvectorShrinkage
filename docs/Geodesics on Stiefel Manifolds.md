# Geodesics on Stiefel Manifolds: A Geometric Picture

## Introduction

The **Stiefel manifold** $V_{n,k}$ consists of all ordered orthonormal $k$-frames in $\mathbb{R}^n$. We represent each frame as an $n \times k$ matrix $X$ with orthonormal columns: $X^T X = I_k$. As a homogeneous space, $V_{n,k} \cong O(n)/O(n-k)$.

## The Tangent Space Structure

At a point $X \in V_{n,k}$, the tangent space decomposes as:

$$
T_X V_{n,k} = \{\dot{X} = XA + X_\perp B : A \in \mathfrak{so}(k), B \in \mathbb{R}^{(n-k) \times k}\}
$$

where:

- $X_\perp$ is an $n \times (n-k)$ matrix whose columns form an orthonormal basis for the orthogonal complement of $\text{span}(X)$, satisfying $X^T X_\perp = 0$ and $X_\perp^T X_\perp = I_{n-k}$
- $A$ is a $k \times k$ **skew-symmetric** matrix representing infinitesimal rotation within the frame
- $B$ is an arbitrary $(n-k) \times k$ matrix representing infinitesimal motion into orthogonal directions

Together, $[X \mid X_\perp]$ forms an $n \times n$ orthogonal matrix.

## The Canonical Metric

The natural $O(n)$-invariant metric is:

$$
g_X(\dot{X}, \dot{X}) = \text{tr}(\dot{X}^T \dot{X}) = \text{tr}(A^T A) + \text{tr}(B^T B)
$$

## Geodesic Formula

A geodesic starting at $X_0$ with initial velocity $\dot{X}_0 = X_0 A + X_{\perp,0} B$ is given by:

$$
X(t) = R(t) \cdot X_0
$$

where $R(t) = \exp(t\Omega)$ is the one-parameter rotation subgroup of $O(n)$ defined by:

$$
\Omega = \begin{bmatrix} A & -B^T \\ B & 0_{(n-k) \times (n-k)} \end{bmatrix}
$$

Equivalently:

$$
X(t) = \begin{bmatrix} X_0 & X_{\perp,0} \end{bmatrix} \exp(t\Omega) \begin{bmatrix} I_k \\ 0 \end{bmatrix}
$$

**Key insight:** The geodesic on the Stiefel manifold is the orbit of the initial frame $X_0$ under a rotation $R(t) \in O(n)$.

## Geometric Interpretation

A geodesic exhibits two types of motion:

1. **Internal rotation** (from $A$): The vectors within the frame rotate relative to each other
2. **Tilting** (from $B$): The entire $k$-dimensional subspace spanned by the frame rotates into orthogonal directions

These motions are **coupled** through the matrix exponential $\exp(t\Omega)$.

## Example: Pure Tilting in $\mathbb{R}^3$

Consider $V_{3,2}$ (2-frames in $\mathbb{R}^3$) starting at:

$$
X_0 = \begin{bmatrix} 1 & 0 \\ 0 &  1 \\  0 & 0 \end{bmatrix}, \quad X_{\perp,0} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

Choose initial velocity with $A = 0$ (no internal rotation) and:

$$
B = \begin{bmatrix} 1 & 0 \end{bmatrix}
$$

This gives:

$$
\Omega = \begin{bmatrix} 0 & 0 & -1 \\ 0 & 0 & 0 \\  1 & 0 & 0 \end{bmatrix}
$$

The rotation is:

$$
R(t) = \exp(t\Omega) = \begin{bmatrix} \cos t & 0 & -\sin t \\ 0 & 1 & 0 \\  \sin t & 0 & \cos t \end{bmatrix}
$$

This is **rotation about the $y$-axis**.

The geodesic on the Stiefel manifold is:

$$
X(t) = R(t) X_0 = \begin{bmatrix} \cos t & 0 \\ 0 & 1 \\  \sin t & 0 \end{bmatrix}
$$

### Visual Description

The two column vectors evolve as:

- **First vector:** $v_1(t) = (\cos t, 0, \sin t)$ rotates in the $xz$-plane, tilting from the $x$-axis toward the $z$-axis
- **Second vector:** $v_2(t) = (0, 1, 0)$ remains fixed on the $y$-axis (the rotation axis)

Picture the $xy$-plane **rotating about the $y$-axis like a door on a hinge**. The plane tilts progressively toward the $z$-direction while maintaining its internal structure.

### Verification: This Is Pure Tilting

To verify there's no internal rotation, compute:

$$
A(t) = X(t)^T \dot{X}(t)
$$

We have:

$$
\dot{X}(t) = \begin{bmatrix} -\sin t & 0 \\ 0 &  0 \\  \cos t & 0 \end{bmatrix}
$$

Therefore:

$$
A(t) = \begin{bmatrix} \cos t & 0 & \sin t \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} -\sin t & 0 \\  0 & 0 \\ \ \cos t & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

Since $A(t) = 0$ throughout the geodesic, there is **no internal rotation** at any time. The two vectors maintain their relative orientation within the plane they span — they don't rotate with respect to each other. This is pure tilting.

## Important Note: Geodesics Are Not Rotations

Although the geodesic arises from a rotation $R(t) \in O(n)$, the geodesic $X(t)$ itself is **not a rotation matrix**:

- $X(t)$ is $n \times k$ (rectangular), not $n \times n$ (square)
- $\dot{X}(t) X(t)^T$ is **not skew-symmetric** in general
- $X(t)$ represents a $k$-frame, not a rotation of all of $\mathbb{R}^n$

The geodesic is the **projection** of a rotation curve in $O(n)$ onto the Stiefel manifold.

## Additional Examples

### Example: $V_{3,1} = S^2$ (The Unit Sphere)

For $k=1$, every point is a unit vector and geodesics are great circles:

$$
v(t) = \cos(t) v_0 + \sin(t) w_0
$$

where $v_0$ is the starting point and $w_0$ is the initial velocity direction.

### Example: Pure Internal Rotation

Starting from the same $X_0$ in $V_{3,2}$ with:

$$
A = \begin{bmatrix} 0 & -\omega \\ \omega & 0 \end{bmatrix}, \quad B = 0
$$

The geodesic is:

$$
X(t) = \begin{bmatrix} \cos(\omega t) & -\sin(\omega t) \\ \sin(\omega t) & \cos(\omega t \\  0 & 0 \end{bmatrix}
$$

The frame stays in the $xy$-plane but rotates internally — like the hands of a clock spinning while the clock face remains flat.

### Example: Coupled Motion

When both $A \neq 0$ and $B \neq 0$, the frame simultaneously spins internally AND tilts into orthogonal directions. This produces helical trajectories through the Stiefel manifold.

## Summary

Geodesics on Stiefel manifolds:

1. **Are orbits of rotations:** $X(t) = R(t) X_0$ where $R(t) = \exp(t\Omega) \in O(n)$

2. **Exhibit coupled motion:** Internal rotation and external tilting are linked through $\exp(t\Omega)$

3. **Are complete:** All geodesics exist for all $t \in \mathbb{R}$ because $O(n)$ is compact

4. **Decompose naturally:** The tangent vector splits into vertical (internal rotation) and horizontal (tilting) components

This geometric structure makes Stiefel manifolds fundamental in optimization on manifolds, matrix factorization problems, and applications throughout statistics and machine learning.
