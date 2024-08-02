# Estimate conditional-independence sets in discrete graphical model without positivity constraints
Based on: F. Leonardi, R. Carvalho. Structure recovery for partially observed discrete Markov random fields on graphs under not necessarily positive distributions. 2022.


## Introduction

Given binary matrices $`X,Y`$, where the rows are realizations and the columns binary variables, we estimate the conditional independence $`X_i indep. X_j | X`$. 
This is, for every pair of variables $`X_i,X_j`$, we return 0 if that pair is conditional independent and 1 if they are not conditional independent given $`Y`$ and $`{X_k, 1<=k<=p: k!=i, k!=j}`$.

## Use
```
from discrete_gm_nonpos import discrete_graphical_model
ci=discrete_graphical_model(c=.1,conservative=False).estimate_CI(X, Y),
```
where $`c>0`$ is a regularization constant and conservative reduces the amount of false positives but adds more false negatives. 
