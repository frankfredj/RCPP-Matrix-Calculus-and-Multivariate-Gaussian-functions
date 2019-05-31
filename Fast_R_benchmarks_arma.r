library("microbenchmark")
library("doParallel")
library("tictoc")
library("gmodels")
library("Rcpp")
library("RcppArmadillo")
library("RSpectra")
library("KernelKnn")
library("ClusterR")

#Load cpp code
sourceCpp("C:/Users/Francis/SkyDrive/Documents/Fast_R_arma.cpp")

#enable openmp and c++11
Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
Sys.setenv("PKG_LIBS" = "-fopenmp")
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")


ncores <- detectCores()

#Vector function benchmark
test_matrix <- matrix(nrow = 1000, ncol = 1000, rnorm(mean = 0, sd = 1, n = 1000^2))
microbenchmark(apply(test_matrix, 2, mean), meanCa(test_matrix, ncores))
microbenchmark(apply(test_matrix, 2, var), varCa(test_matrix, ncores))
microbenchmark(apply(test_matrix, 2, sd), sdCa(test_matrix, ncores))


#Test centering and scaling
microbenchmark(scale(test_matrix), center_scaleCa(test_matrix, ncores))

#Test centering and scaling + return cor
microbenchmark(cor(test_matrix), fast_corCa(test_matrix, ncores))




#--------------------------------
#--------------------------------
#Test high-dimention PCA
#--------------------------------
#--------------------------------


data <- matrix(nrow = 10000, ncol = 1000, rnorm(mean = 5, sd = 15, n = 1000^2))

for(i in 1:500){

index_1 <- floor(runif(min = 1, max = ncol(data), n = 1))
index_2 <- floor(runif(min = 1, max = ncol(data), n = 1))

data[,index_1] <- data[,index_1] + runif(min = 0, max = 1, n = 1) * data[,index_2]

}


tic()

#Get the centered + scaled first 150 PC of a 10 000 x 1000 matrix

n_dim <- 50

v_cov <- fast_corCa(data, ncores)
eig <- eigs_sym(v_cov, 150, which = "LM")
reduced_data <- data %*% eig$vectors 
center_scaleCa(reduced_data, ncores)

toc()

#Test knn (i.e.: for clustering large dataset)

km = KMeans_arma(reduced_data, clusters = 30, n_iter = 25, seed_mode = "random_subset", 
                 
                 verbose = T, CENTROIDS = NULL)

#--------------------------------
#--------------------------------
#Test high-dimention PCA
#--------------------------------
#--------------------------------





#Test MVN

L <- matrix(nrow = 1000, ncol = 100, 0)

L[,1] <- rnorm(mean = 0, sd = 0.1, n = 1000)

for(i in 2:ncol(L)){

weights = runif(min = 0, max = 1, n = (i-1))
weights = weights / sum(weights)

for(j in 2:i){

L[,i] <- L[,i] + L[,(j-1)] * weights[j-1]

}

}

L <- 0.5*L + 0.5*matrix(nrow = 1000, ncol = 100, rnorm(mean = 0, sd = 0.1, n = 1000*100))

mu <- apply(L, 2, mean)

L <- chol(var(L))

L_vec <- c(L[upper.tri(L, diag = TRUE)])



library("microbenchmark")
library("MASS")

microbenchmark(mvrnorm(1, mu, L), mvnrv_vec(mu, L_vec, 7))
microbenchmark(mvrnorm(100, mu, L), mvnrv(mu, L_vec, 100, 7))


sample_r <- mvnrv(mu, L_vec, 100, 7)
sample_cpp <-mvrnorm(100, mu, L)

cov_r <- var(sample_r)
cov_cpp <- var(sample_cpp)

mean_r <- apply(sample_r, 2, mean)
mean_cpp <- apply(sample_cpp, 2, mean)

print(mean(abs(cov_r - cov_cpp)))
print(mean(abs(mean_r - mean_cpp)))

print(mean(abs(L - cov_cpp)))
print(mean(abs(mu - mean_cpp)))




#Test truncated MVN

above <- mu -0.1
below <- mu + 0.1

library("tmvnsim")

sigma = t(L) %*% L

microbenchmark(tmvnsim(100, ncol(L), sigma = sigma, lower = above, upper = below, means = mu),
TruncMvnrv(mu, L_vec, above, below, 100, 7)
	)


sample_r <- tmvnsim(100, ncol(L), sigma = sigma, lower = above, upper = below, means = mu)$samp
sample_cpp <- TruncMvnrv(mu, L_vec, above, below, 100, 7)

cov_r <- var(sample_r)
cov_cpp <- var(sample_cpp)

mean_r <- apply(sample_r, 2, mean)
mean_cpp <- apply(sample_cpp, 2, mean)

print(mean(abs(cov_r - cov_cpp)))
print(mean(abs(mean_r - mean_cpp)))


#Test one-sided truncated

is_above <- sample(size = 100, c(0,1), replace = TRUE)
trumpt <- c(1:100)
trumpt[which(is_above == 1)] <- above[which(is_above == 1)]
trumpt[which(is_above == 0)] <- above[which(is_above == 0)]

microbenchmark(tmvnsim(100, ncol(L), sigma = sigma, upper = below, means = mu),
OneSideTruncMvnrv(mu, L_vec, is_above, trumpt, 100, 7)
	)

upper <- rep(1,100)
trumpt <- below

sample_r <- tmvnsim(100, ncol(L), sigma = sigma, upper = below, means = mu)$samp
sample_cpp <- OneSideTruncMvnrv(mu, L_vec, upper, below, 100, 7)


cov_r <- var(sample_r)
cov_cpp <- var(sample_cpp)

mean_r <- apply(sample_r, 2, mean)
mean_cpp <- apply(sample_cpp, 2, mean)

print(mean(abs(cov_r - cov_cpp)))
print(mean(abs(mean_r - mean_cpp)))




#Test ghk integrals

microbenchmark(ghk_two_side(L_vec, above, below, 50, 7), mean(tmvnsim(50, ncol(L), sigma = sigma, upper = below, lower = above)$wts))

ghk_tmv <- rep(0,100)
ghk_rcpp <- rep(0,100)

for(i in 1:100){

ghk_tmv[i] <- mean(tmvnsim(50, ncol(L), sigma = sigma, upper = below, lower = above)$wts)
ghk_rcpp[i] <- ghk_two_side(L_vec, above, below, 50, 7)

}

print("abs diff / mean(tvm")
abs(mean(ghk_tmv) - mean(ghk_rcpp)) / mean(ghk_tmv)
print("var rcpp / var tmv")
var(ghk_rcpp) / var(ghk_tmv)



microbenchmark(ghk_oneside(L_vec, below, upper, 50, 7), mean(tmvnsim(50, ncol(L), sigma = sigma, upper = below)$wts))

ghk_tmv <- rep(0,100)
ghk_rcpp <- rep(0,100)

for(i in 1:100){

ghk_tmv[i] <- mean(tmvnsim(50, ncol(L), sigma = sigma, upper = below)$wts)
ghk_rcpp[i] <- ghk_oneside(L_vec, below, upper, 50, 7)

}

print("abs diff / mean(tvm")
abs(mean(ghk_tmv) - mean(ghk_rcpp)) / mean(ghk_tmv)
print("var rcpp / var tmv")
var(ghk_rcpp) / var(ghk_tmv)