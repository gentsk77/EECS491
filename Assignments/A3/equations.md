Started with the original energy function equation $E$, we have: 

$$ E(\mathbf{x}, \mathbf{y}) = h \sum_i x_i - \beta \sum_{i,j} x_i x_j - \eta \sum_i x_i y_i $$

  Now let $x_k'$ denote the state of $x_k$ after changing, and $E'$ denotes the new energy equation.
  Since each pixel of the image we're discussing here is a binary variable, we may simply assume that the change of one variable is either from $+1$ to $-1$ or vice-versa. By flipping the state of $x_i$, we simply flip by sign and thus obtain $x_i' = -x_i$.

  Therefore, to calculate the difference in the energy equation, we simply substract $E'$ from $E$ as below:

  $$
  E' - E =    E(\bold{x}', \bold{y}) -   E(\bold{x}, \bold{y})  
  \newline =  h * (x_-1 + ... + x_k' + ... + x_n) - \beta * (x_-1x_1 + ... + x_-1x_k' + ... + x_-1x_n + ... + x_k'x_-1 + ... + x_k'x_n + ... 
  \newline + x_nx_-1 + ... + x_nx_k' + ... + x_nx_{n-1}) - \eta * (x_-1y_-1 + ... + x_k'y_k + ... + x_ny_n)
  \newline - (h * (x_-1 + ... + x_k + ... + x_n) - \beta * (x_-1x_1 + ... + x_-1x_k + ... + x_-1x_n + ... + x_kx_-1 + ... + x_kx_n + ... 
  \newline + x_nx_-1 + ... + x_nx_k + ... + x_nx_{n-1}) - \eta * (x_-1y_-1 + ... + x_ky_k + ... + x_ny_n))
  \newline = h * (x_-1 + ... - x_k + ... + x_n) - \beta * (x_-1x_1 + ... - x_-1x_k + ... + x_-1x_n + ... - x_kx_-1 + ... - x_kx_n + ... 
  \newline + x_nx_-1 + ... - x_nx_k + ... + x_nx_{n-1}) - \eta * (x_-1y_-1 + ... - x_ky_k + ... + x_ny_n)
  \newline - (h * (x_-1 + ... + x_k + ... + x_n) - \beta * (x_-1x_1 + ... + x_-1x_k + ... + x_-1x_n + ... + x_kx_-1 + ... + x_kx_n + ... 
  \newline + x_nx_-1 + ... + x_nx_k + ... + x_nx_{n-1}) - \eta * (x_-1y_-1 + ... + x_ky_k + ... + x_ny_n))
  \newline E' - E = -2hx_k + 2\beta (2 \sum_i x_i x_k) + 2 \eta x_k y_k = -2hx_k + 4\beta \sum_i x_i x_k + 2 \eta x_k y_k
  $$

And the final equation $E' - E$ that specifies the change in the energy equation when one variable $x_k$ changes state is:

$$ E' - E = -2hx_k + 4\beta \sum_i x_i x_k + 2 \eta x_k y_k $$

$$ 
 E(\mathbf{x}, \mathbf{y}) = h \sum_i x_i + \beta \sum_{i, j} |x_i - x_j| + \eta \sum_i |x_i - y_i|
$$

$$
E' - E = h(x_k' - x_k) + 2 \beta \sum_{j \in nbr(k)} (|x_k' - x_j| - |x_k - x_j|) + \eta (|x_k' - y_k| - |x_k - y_k|)
$$

$$
p(X) = \frac{1}{Z} \psi_{a,b,d} (a, b, d) \psi_{a, c}(a, c) \psi_{c, f}(c, f) \psi_{b, e}(b, e) \psi_{d, g}(d, g) \psi_{d, h}(d, h)
$$


leaves $\rightarrow$ root:

$$
\mu_{d \rightarrow f_2}(d) = 1
$$
$$
\mu_{f_2 \rightarrow c}(d) = \sum_d f_2(d, c) 
$$
$$
\mu_{c \rightarrow f_1}(c) = \mu_{f_2 \rightarrow c}(d)
$$
$$
\mu_{b \rightarrow f_1}(b) = 1
$$

$$
\mu_{f_1 \rightarrow a}(b, c) = \sum_{b, c} f_1(b, c) \mu_{b \rightarrow f_1}(b) \mu_{c \rightarrow f_1}(c)
$$



$$
\mu_{a \rightarrow f_1}(a) = 1
$$

$$
\mu_{f_1 \rightarrow c}(a, b) = \sum_{a, b} f_1(a, b, c) \mu_{b \rightarrow f_1}(b) \mu_{a \rightarrow f_1}(a)
$$

$$
\mu_{c \rightarrow f_2}(c) = \mu_{f_1 \rightarrow c}(a, b)
$$

$$
\mu_{f_2 \rightarrow d}(c) = \sum_c f_2(c, d) \mu_{c \rightarrow f_2}(c)
$$