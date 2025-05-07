# Estimate conditional-independence sets in discrete graphical model without positivity constraints
Based on: F. Leonardi, R. Carvalho. Structure recovery for partially observed discrete Markov random fields on graphs under not necessarily positive distributions. 2022.


## Introduction

Given binary matrices $`X,Y`$, where the rows are realizations and the columns binary variables, we estimate the conditional independence $`X_i\, \text{indep.}\, X_j \,|\, Y, X_{\setminus \{i,j\}}`$. 
This is, for every pair of variables $`X_i,X_j`$, the adjacency matrix is encoded so that an entry is $`0`$ if the pair of variables is conditional independent and $`1`$ if they are not conditional independent given $`Y`$ and $`X_{\setminus \{i,j\}}`$.

## Use
```
from discrete_gm_nonpos import discrete_graphical_model

dgm = discrete_graphical_model(c=np.geomspace(1e-9, 1e3,10000), ncores=4)

CI  = dgm.estimate_stable_CI(X, Y, PFER=1, npartitions=100, seed=None)
```
where $`c_i>0`$ is a penalization constant.

CI returns a dictionary with keys 'conserv' and 'nconserv', each one contains an estimated adjacency matrix with per-family error rate (PFER) controlled. 

When running multiple datasets with the same parameters, use 

```
CI_multiple = dgm.estimate_stable_CI_multiple_datasets([(X1,Y1),(X2,Y2),(X3,Y3)], ncores_outer=3, PFER=1, npartitions=100, seed=None)

```
which return a list of dictionaries.


## Example

```
import numpy as np
from scipy.stats import norm
from discrete_gm_nonpos import discrete_graphical_model
if __name__ == "__main__": # test
    n_samples = 100
    
    # Correlation matrix (example, must be positive semi-definite)
    correlation_matrix = np.array([
        [1.0, 0.8, 0.0, 0.0, 0.0],  # First variable
        [0.8, 1.0, 0.8, 0.0, 0.0],  # Second variable
        [0.0, 0.8, 1.0, 0.0, 0.0],  # Third variable
        [0.0, 0.0, 0.0, 1.0, 0.9],  # Fourth variable
        [0.0, 0.0, 0.0, 0.9, 1.0]   # Fifth variable
    ])
    
    # Generate correlated normal samples
    normal_samples = np.random.multivariate_normal(np.zeros(correlation_matrix.shape[0]), correlation_matrix, size=n_samples)
    
    # Convert to binary using the normal CDF thresholding
    binary_samples = (norm.cdf(normal_samples) > 0.5).astype(int)
    
    Y = binary_samples[:,0,None]
    X = binary_samples[:,1:]
    
    dgm = discrete_graphical_model(np.geomspace(1e3, 1e-3,1000),ncores=4)
    
    #cihat = dgm.estimate_CI(X>0, Y>0)
    
    CI_stable =dgm.estimate_stable_CI(X,Y,PFER=2,npartitions=100,seed=None)
    
    # {'conserv': array([[False, False, False, False],
    #     [False, False, False, False],
    #     [False, False, False,  True],
    #     [False, False,  True, False]]),
    # 'nconserv': array([[False, False, False, False],
    #     [False, False, False, False],
    #     [False, False, False,  True],
    #     [False, False,  True, False]])}
    
    CI_multiple = dgm.estimate_stable_CI_multiple_datasets([(X,Y),(X,Y),(X,Y)],ncores_outer=3, PFER=2, npartitions=100, seed=None)
    
    #  [{'conserv': array([[False, False, False, False],
    #       [False, False, False, False],
    #       [False, False, False,  True],
    #       [False, False,  True, False]]),
    #  'nconserv': array([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False,  True],
    #         [False, False,  True, False]])},
    # {'conserv': array([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False,  True],
    #         [False, False,  True, False]]),
    #  'nconserv': array([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False,  True],
    #         [False, False,  True, False]])},
    # {'conserv': array([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False,  True],
    #         [False, False,  True, False]]),
    #  'nconserv': array([[False, False, False, False],
    #         [False, False, False, False],
    #         [False, False, False,  True],
    #         [False, False,  True, False]])}]
```