
$$
(\lambda - var(y) v_2) (\lambda - var(x)v_1) = \Sigma_{yx} \Sigma_{xy} 
$$

$$
(\lambda - var(x)v_1) = \frac {\Sigma_{yx} \Sigma_{xy}}{(\lambda - var(y) v_2) } 
$$

$$
var(x)v_1 = \lambda - \frac {\Sigma_{yx} \Sigma_{xy}}{(\lambda - var(y) v_2) } 
$$

$$
v_1 = \frac{\lambda^2 - \lambda var(y) v_2 - \Sigma_{yx} \Sigma_{xy}} {var(x) (\lambda - var(y) v_2)}
$$

$$
v_2 = 
$$



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
\Sigma_{yx}v_1 = (\lambda - var(y) v_2)v_2
$$



$$
\Sigma_{xy} v_2 = \lambda v_1 - var(x)v_1^2
$$



$$
v_2 = \frac{\lambda v_1 - var(x)v_1^2}{\Sigma_{xy}}
$$

$$
v_1 = \frac{\lambda v_2 - var(y)v_2^2}{\Sigma_{yx}}
$$


$$
v_1 = \frac{\lambda \frac{\lambda v_1 - var(x)v_1^2}{\Sigma_{xy}} - var(y)( \frac{\lambda v_1 - var(x)v_1^2}{\Sigma_{xy}})^2}{\Sigma_{yx}}
$$



$$
\Sigma - \lambda I = \begin{bmatrix} var(x) \space \space \space \space \Sigma_{x y} \\ \Sigma_{y x} \space \space \space \space var(y) \end{bmatrix} - \begin{bmatrix} \lambda \space \space 0 \\ 0 \space \space \lambda  \end{bmatrix} = \begin{bmatrix} var(x) - \lambda \space \space \space \space \space \space \space \space \space \space \Sigma_{xy} \\ \Sigma_{yx} \space \space \space \space \space \space \space \space \space \space var(y) - \lambda  \end{bmatrix}
$$


Then we use the determinant to solve for the eigenvalue:

$$
var(x)var(y) + \lambda ^2 - \lambda var(x) - \lambda var(y) = \Sigma_{xy} \Sigma_{yx}
$$


$$
\lambda ^2 - ( var(x) + var(y)) \lambda - \Sigma_{xy} \Sigma_{yx} + var(x)var(y) = 0
$$

$$
\lambda_1 = \frac{var(x) + var(y) + \sqrt{( var(x) - var(y))^2 - 4 \Sigma_{xy} \Sigma_{yx}}}{2}
$$

$$
\lambda_2 = \frac{var(x) + var(y) - \sqrt{( var(x) - var(y))^2 - 4\Sigma_{xy} \Sigma_{yx}}}{2}
$$














