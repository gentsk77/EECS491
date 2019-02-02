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




Vanderwal took care of Huhu for me over the past winter break when I wasn't home. To ensure that our cats were not trying to kill each other when he's not home, he installed a petcam in his living room to spy on the kittens. Vanderwal left the petcam on no matter he's home or not so I can also take a look at Huhu when I miss him.  
One major function of the petcam is to spy on motions: when the kittens are chasing around, my iPhone app will send me a notification, so that I could know there was a fight going on. In order to model the probability of kitten fight, let's have two more **independent** variables based on my observation along with the probability I'd like to model as below:  

- `F`: the probability that the kittens are fighting each other. 
- `V`: Vanderwal is home. Let $p(v)$ denotes $p(Vanderwal~is~Home=True)$
- `N`: the kittens have just finished a nap. Let $p(n)$ denotes $p(Napped=True)$
  
With the above varaibles, we shall model the posterior distribution of the probability that the kittens are having a fight using Bayes rule: 
$$p(f|V,N) = \frac{p(V, N|f) * p(f)}{p(V,N)} = p(f | V, C) = \frac{(p(V|f) * P(C|s) * p(f)}{P(L|s) * P(C|s) * p(f) + P(L|\bar{s}) * P(C|\bar{s}) * P(\bar{s})}$$
