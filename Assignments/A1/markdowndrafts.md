|      | $p(B)$ |
| :--- | :----- |
| T    | 0.3    |
| F    | 0.7    |


|      | $p(H)$ |
| :--- | :----- |
| T    | 0.15   |
| F    | 0.85   |


| $B$  | $p(P\mid B)$ |
| :--- | :----------- |
| T    | 0.9          |
| F    | 0.01         |

| $H$  | $p(S\mid H)$ |
| :--- | :----------- |
| T    | 0.6          |
| F    | 0.25         |

| $B$  | $H$  | $p(U\mid B,H)$ |
| :--- | :--- | :------------- |
| T    | T    | 0.8            |
| T    | F    | 0.2            |
| F    | T    | 0.05           |
| F    | F    | 0.01           |


  In the previous part, we have derived a posterior distribution equation assuming a Gamma prior for the Poisson distribution.  
  We can easily conjugate the expression to what we want in this part as below:
  $$ p(\lambda |n,T,\alpha,\beta) = \frac {exp(-\lambda T)(\lambda T)^{n}}{n!} \frac {\beta ^ {\alpha}}{\Gamma (\alpha)} \lambda ^{\alpha - 1}exp(-\beta \lambda) $$
  $$ p(\lambda |n,T,\alpha,\beta) = \frac {T^{n}}{n!} \frac {\beta ^ {\alpha}}{\Gamma (\alpha)} \lambda ^{\alpha - 1 + n}exp(-\beta \lambda - T \lambda) $$
  $$ p(\lambda |n,T,\alpha,\beta) = \frac {T^{n}}{n!} \frac {\beta ^ {\alpha}}{\Gamma (\alpha)} \lambda ^{\alpha + n - 1}exp(-(\beta + T) \lambda) $$