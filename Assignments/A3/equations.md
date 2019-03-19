Started with the original energy function equation $E$, we have: 

$$ E(\mathbf{x}, \mathbf{y}) = h \sum_i x_i - \beta \sum_{i,j} x_i x_j - \eta \sum_i x_i y_i $$

  Now let $x_k'$ denote the state of $x_k$ after changing, and $E'$ denotes the new energy equation.
  Since each pixel of the image we're discussing here is a binary variable, we may simply assume that the change of one variable is either from $+1$ to $-1$ or vice-versa. By flipping the state of $x_i$, we simply flip by sign and thus obtain $x_i' = -x_i$.

  Therefore, to calculate the difference in the energy equation, we simply substract $E'$ from $E$ as below:

  $$
  E' - E =    E(\bold{x}', \bold{y}) -   E(\bold{x}, \bold{y})  
  \newline =  h * (x_0 + ... + x_k' + ... + x_n) - \beta * (x_0x_1 + ... + x_0x_k' + ... + x_0x_n + ... + x_k'x_0 + ... + x_k'x_n + ... 
  \newline + x_nx_0 + ... + x_nx_k' + ... + x_nx_{n-1}) - \eta * (x_0y_0 + ... + x_k'y_k + ... + x_ny_n)
  \newline - (h * (x_0 + ... + x_k + ... + x_n) - \beta * (x_0x_1 + ... + x_0x_k + ... + x_0x_n + ... + x_kx_0 + ... + x_kx_n + ... 
  \newline + x_nx_0 + ... + x_nx_k + ... + x_nx_{n-1}) - \eta * (x_0y_0 + ... + x_ky_k + ... + x_ny_n))
  \newline = h * (x_0 + ... - x_k + ... + x_n) - \beta * (x_0x_1 + ... - x_0x_k + ... + x_0x_n + ... - x_kx_0 + ... - x_kx_n + ... 
  \newline + x_nx_0 + ... - x_nx_k + ... + x_nx_{n-1}) - \eta * (x_0y_0 + ... - x_ky_k + ... + x_ny_n)
  \newline - (h * (x_0 + ... + x_k + ... + x_n) - \beta * (x_0x_1 + ... + x_0x_k + ... + x_0x_n + ... + x_kx_0 + ... + x_kx_n + ... 
  \newline + x_nx_0 + ... + x_nx_k + ... + x_nx_{n-1}) - \eta * (x_0y_0 + ... + x_ky_k + ... + x_ny_n))
  \newline E' - E = -2hx_k + 2\beta (2 \sum_i x_i x_k) + 2 \eta x_k y_k = -2hx_k + 4\beta \sum_i x_i x_k + 2 \eta x_k y_k
  $$

And the final equation $E' - E$ that specifies the change in the energy equation when one variable $x_k$ changes state is:

$$ E' - E = -2hx_k + 4\beta \sum_i x_i x_k + 2 \eta x_k y_k $$