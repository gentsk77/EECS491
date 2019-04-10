### $x$ and $y$ are uncorrelated

Since $x$ and $y$ are uncorrelated, we may conclude that $\Sigma_{x y} = \Sigma_{y x} = 0$, and thus the convariance matrix should be defined as:

$$  
\Sigma =  \begin{bmatrix} var(x) \space \space \space \space 0 \\ 0 \space \space \space \space var(y) \end{bmatrix}
$$

### $x$ and $y$ are correlated

Since $x$ and $y$ are positively correlated, we may conclude that $\Sigma_{x y} > 0$, $\Sigma_{y x} > 0$, and thus the convariance matrix should be defined as:

$$  
\Sigma =  \begin{bmatrix} var(x) \space \space \space \space \Sigma_{x y} \\ \Sigma_{y x} \space \space \space \space var(y) \end{bmatrix}, \Sigma_{x y} > 0, \Sigma_{y x} > 0
$$

### $x$ and $y$ are anti-correlated

Since $x$ and $y$ are negatively correlated, we may conclude that $\Sigma_{x y} < 0$, $\Sigma_{y x} < 0$, and thus the convariance matrix should be defined as:

$$  
\Sigma =  \begin{bmatrix} var(x) \space \space \space \space \Sigma_{x y} \\ \Sigma_{y x} \space \space \space \space var(y) \end{bmatrix}, \Sigma_{x y} < 0, \Sigma_{y x} < 0
$$






$$
\begin{bmatrix} var(x) \space \space \space \space 0 \\ 0 \space \space \space \space var(y) \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} \lambda v_1 \\ \lambda v_2 \end{bmatrix}
$$

$$
\begin{bmatrix} var(x) v_1 \\ var(y) v_2 \end{bmatrix}  = \begin{bmatrix} \lambda v_1 \\ \lambda v_2 \end{bmatrix}
$$


$$  
\begin{bmatrix} var(x) \space \space \space \space \Sigma_{x y} \\ \Sigma_{y x} \space \space \space \space var(y) \end{bmatrix}  \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} \lambda v_1 \\ \lambda v_2 \end{bmatrix}
$$

$$
\begin{bmatrix} var(x) v_1 + \Sigma_{xy} v_2 \\ \Sigma_{yx} v_1 + var(y) v_2 \end{bmatrix}  = \begin{bmatrix} \lambda v_1 \\ \lambda v_2 \end{bmatrix}
$$

$$
\Sigma_{xy} v_2 = (\lambda - var(x)v_1)v_1
$$

$$
\Sigma_{yx}v_1 = (\lambda - var(y))v_2
$$

$$
\Sigma_{yx}v_1 = (\lambda - var(y)) (\lambda - var(x)v_1)v_1 / \Sigma_{xy}
$$