## Comparison (ex. $D_k = 3$, w/o positional encoding)

|          | A: `weights=True`, `bias=True`           | B: `weights=False`, `bias=False`, $[1,\dots,1]^T$ | C: `weights=False`, `bias=False`, identity   |
|--------------|-------------------------------------------|---------------------------------------------------------|----------------------------------------------------|
| $x_\ell$     | $\mathbb{R}^3$                            | $\mathbb{R}^3$                                          | $\mathbb{R}^3$                                    |
| $W_q$        | $\mathbb{R}^{3 \times 3}$           | $\mathbb{R}^{1 \times 3}$, $[1,\dots,1]^T$                  | $\mathbb{R}^{3 \times 3}$, identity matrix          |
| $b_q$        | $\mathbb{R}^3$                      | $\mathbf{0}=[0,0,0]^T \in \mathbb{R}^3$                 | $\mathbf{0}=[0,0,0]^T \in \mathbb{R}^3$           |
| $q_\ell$     | $\mathbb{R}^3$                            | $\mathbb{R}^3$                          | $\mathbb{R}^3$                                    |

## Example

$$
x_\ell = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix} \in \mathbb{R}^3
$$


### A. Trainable Weight and Bias

$$
W_q = \begin{bmatrix}
1 & 0 & -1 \\
0 & 1 & 0 \\
2 & -1 & 1
\end{bmatrix}, \quad
b_q = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}
$$

$$
q_\ell = W_q x_\ell + b_q =
\begin{bmatrix}
-2.0 \\ 2.0 \\ 3.0
\end{bmatrix}
+ \begin{bmatrix}
0.1 \\ 0.2 \\ 0.3
\end{bmatrix}
=
\begin{bmatrix}
-1.9 \\ 2.2 \\ 3.3
\end{bmatrix}
$$


### B. weight = $[1, 1, 1]^T$

$$
W_q = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
$$


### C. weight = identity

$$
W_q = I_3 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
q_\ell = W_q x_\ell  + b_q = x_\ell + b_q= \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix}
$$
