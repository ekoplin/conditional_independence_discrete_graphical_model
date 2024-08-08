# Estimate conditional-independence sets in discrete graphical model without positivity constraints
Based on: F. Leonardi, R. Carvalho. Structure recovery for partially observed discrete Markov random fields on graphs under not necessarily positive distributions. 2022.


## Introduction

Given binary matrices $`X,Y`$, where the rows are realizations and the columns binary variables, we estimate the conditional independence $`X_i\, \text{indep.}\, X_j \,|\, Y, X_{\setminus \{i,j\}}`$. 
This is, for every pair of variables $`X_i,X_j`$, the adjacency matrix is encoded so that an entry is $`0`$ if the pair of variables is conditional independent and $`1`$ if they are not conditional independent given $`Y`$ and $`X_{\setminus \{i,j\}}`$.

## Use
```
from discrete_gm_nonpos import discrete_graphical_model
CI=discrete_graphical_model(c=np.lispace(.1,1,10),ncores=None).estimate_CI(X, Y)
```
where $`c_i>0`$ is a penalization constant.

CI returns a dictionary with keys 'conserv' and 'nconserv', each value is a list containing the adjacency matrices corresponding to each value of $c_i$.
