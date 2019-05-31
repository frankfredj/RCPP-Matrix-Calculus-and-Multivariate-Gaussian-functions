# Matrix calculus

Benchmarked on a 1000 x 1000 matrix filled with Gaussian(0,1) random variables

Specs: i3630qm processor (2012)

## Mean, Variance and Standard Deviation

Parallelised versions of R's mean, var and sd using c++ 's accumulate. 
![](https://i.imgur.com/uHeIRZR.png)

## Covariance and Correlation

Parallelised versions of R's cov and cor. 

![](https://i.imgur.com/FeEh1nP.png)

## Centering and Scaling

Parallelised version of R's scale using pointers.

![](https://i.imgur.com/M15AWQ0.png)



# Multivariate Gaussian functions

Note that the Cholesky decomposition of the variance-covariance matrix needs to be stored inside a vector, i.e.:

```R
L <- chol(var(some_matrix))

L_vec <- c(L[upper.tri(L, diag = TRUE)])
```

All benchmarks were compared to the already avaible package tmvnsim

## Simulating mvn variables

![](https://i.imgur.com/47HjkWL.png)

## Simulating truncated mvn variables

![](https://i.imgur.com/9XXnQOB.png)

## Simulating truncated (one-sided) mvn variables

![](https://i.imgur.com/suc6V15.png)

Note that the is_above should contain 1 if the truncation point (trumpt) is an upper bound, and 0 otherwise.

## Calculating one and two-sided multivariate Gaussian integrals

![](https://i.imgur.com/tMfcNWn.png)








