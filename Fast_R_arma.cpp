#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <omp.h>
#include <cmath>
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::plugins(cpp11)]]

using namespace arma;


#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>


inline static double sqrt_double( double x ){ return ::sqrt( x ); }

// [[Rcpp::export]]
vec meanCa(mat & X, int ncores){

	int ncols = X.n_cols;
	int nrows = X.n_rows;

	vec mu(ncols);

	omp_set_num_threads(ncores);
	#pragma omp parallel for

		for(int j = 0; j < ncols; j++){

			mu(j) = accu(X.col(j)) / nrows;

		}

	return mu;

}



// [[Rcpp::export]]
void scale_vectorCa(vec & v, double & sigma){

std::transform(v.begin(), v.end(), v.begin(), [&] (double x) {return x / sigma;});

}



// [[Rcpp::export]]
void scale_matCa(mat & X, double & sigma, int ncores){

	int ncols = X.n_cols;

	omp_set_num_threads(ncores);
	#pragma omp parallel for

	for(int j = 0; j < ncols; j++){

		std::transform(X.col(j).begin(), X.col(j).end(), X.col(j).begin(), [&] (double x){return x / sigma;});

	}

}



// [[Rcpp::export]]
void sqrt_vectorCa(vec & v){

std::transform(v.begin(), v.end(), v.begin(), [] (double x) {return sqrt(x);});

}



// [[Rcpp::export]]
vec varCa(mat & X, int ncores){

	int ncols = X.n_cols;
	int nrows = X.n_rows;

	vec mu = meanCa(X, ncores);
	vec mu_sq(ncols);

	omp_set_num_threads(ncores);
	#pragma omp parallel for

	for(int j = 0; j < ncols; j++){

		mu_sq(j) = std::accumulate(X.col(j).begin(), X.col(j).end(), 0.0, 

			[] (double & acc, double & x) {return acc + x*x;}) / nrows;

		mu_sq(j) -= mu(j)*mu(j);
	}

	double sc = double(nrows - 1) / double(nrows);

	scale_vectorCa(mu_sq, sc);

	return mu_sq;

}





// [[Rcpp::export]]
vec sdCa(mat & X, int ncores){

	int ncols = X.n_cols;
	int nrows = X.n_rows;

	vec mu = meanCa(X, ncores);
	vec mu_sq(ncols);

	omp_set_num_threads(ncores);
	#pragma omp parallel for

	for(int j = 0; j < ncols; j++){

		mu_sq(j) = std::accumulate(X.col(j).begin(), X.col(j).end(), 0.0, 

			[] (double & acc, double & x) {return acc + x*x;}) / nrows;

		mu_sq(j) -= mu(j)*mu(j);

	}

	double sc = double(nrows - 1) / double(nrows);

	scale_vectorCa(mu_sq, sc);
	sqrt_vectorCa(mu_sq);

	return mu_sq;


}




// [[Rcpp::export]]
void center_scaleCa(mat & X, int ncores){

int ncols = X.n_cols;
int nrows = X.n_rows;

omp_set_num_threads(ncores);
#pragma omp parallel for

for(int j = 0; j < ncols; j++){

X.col(j) -= accu(X.col(j)) / nrows;

X.col(j) /= std::sqrt(std::accumulate(X.col(j).begin(), X.col(j).end(), 0.0, 
			[] (double & acc, double & x) {return acc + x*x;}) / (nrows-1));

}

}




// [[Rcpp::export]]
mat fast_corCa(mat & X, int ncores){

center_scaleCa(X, ncores);

int ncols = X.n_cols;

mat vcov_mat = zeros(ncols, ncols);

omp_set_num_threads(ncores);
#pragma omp parallel for

for(int j = 0; j < ncols; j++){

	for(int i = 0; i < j; i++){

		vcov_mat(i,j) += dot(X.col(i), X.col(j));
		vcov_mat(j,i) += vcov_mat(i,j);

	}

}

#pragma omp parallel for

for(int j = 0; j < ncols; j++){

	vcov_mat(j,j) += dot(X.col(j), X.col(j));

}

double sc = double(X.n_rows - 1);
scale_matCa(vcov_mat, sc, ncores);

return vcov_mat;

}






// [[Rcpp::export]]
double RationalApproximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) / 
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}



// [[Rcpp::export]]
double NormalCDFInverse(double p)
{

    // See article above for explanation of this section.
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}


// [[Rcpp::export]]
double normalCDF(double x) // Phi(-∞, x) aka N(x)
{
    return std::erfc(-x/std::sqrt(2))/2;
}



// [[Rcpp::export]]
double runif_cpp() {

return ((double) rand() / (RAND_MAX));

}



// [[Rcpp::export]]
double rnorm_cpp() {

double u = runif_cpp();

while(u < 0.000001 || u > 0.99999){u = runif_cpp();};

return NormalCDFInverse(u);

}





// [[Rcpp::export]]
vec mvnrv_vec(vec & mu, vec & L_vec, int & ncores){

int dim = mu.n_elem;

vec out = zeros(dim);
vec z(dim);

int from, to;

std::transform(z.begin(), z.end(), z.begin(), [&] (double x) {return rnorm_cpp();});

omp_set_num_threads(ncores);
#pragma omp parallel for

for(int i = 0; i < dim; i++){

from = i * (i + 1) / 2;
to = from + i + 1;

int count = 0;

for(int j = from; j < to; j++){

out[i] += L_vec[j] * z[count]; 

count += 1;

}

out[i] += mu[i];

}


return out;

}



// [[Rcpp::export]]
void mvnrv_single(vec & mu, vec & L_vec, int & k, mat & M){

int dim = mu.n_elem;

vec z(dim);

int from, to;

std::transform(z.begin(), z.end(), z.begin(), [&] (double x) {return rnorm_cpp();});

for(int i = 0; i < dim; i++){

from = i * (i + 1) / 2;
to = from + i + 1;

int count = 0;

for(int j = from; j < to; j++){

M(i,k) += L_vec[j] * z[count]; 

count += 1;

}

}

M.col(k) += mu;

}




// [[Rcpp::export]]
mat mvnrv(vec & mu, vec & L_vec, int n, int & ncores){

int dim = mu.n_elem;

mat M = zeros(dim, n);

omp_set_num_threads(ncores);
#pragma omp parallel for
for(int k = 0; k < n; k++){mvnrv_single(mu, L_vec, k, M);};

return M;

}





// [[Rcpp::export]] 
void TruncMvnrv_single(vec & mu, vec & L_vec, int & k, vec & above, vec & below, mat & M, int & dim){

double sum, Fa, Fb, u;

vec z = zeros(dim);

double from, to, res, count;

for(int i = 0; i < dim; i++){

sum = mu(i);
count = 0.0;

from = i * (i + 1) / 2;
to = from + i + 1;

for(int j = from; j < to-1; j++){

res = L_vec[j] * z[count]; 
sum += res;
M(i,k) += res;
count += 1;

}

Fa = normalCDF((above(i) - sum) / L_vec[to-1]);
Fb = normalCDF((below(i) - sum) / L_vec[to-1]);

u = runif_cpp();

z(i) = NormalCDFInverse(u*Fa + (1-u)*Fb);

M(i,k) += L_vec[to-1] * z[count]; 

}

}


// [[Rcpp::export]] 
mat TruncMvnrv(vec & mu, vec & L_vec, vec & above, vec & below, int & n, int & ncores){

int dim = above.n_elem;

mat M = zeros(dim, n);

omp_set_num_threads(ncores);
#pragma omp parallel for
for(int k = 0; k < n; k++){TruncMvnrv_single(mu, L_vec, k, above, below, M, dim);};

return M;

}




// [[Rcpp::export]] 
void TruncMvnrv_oneside_single(vec & mu, vec & L_vec, int & k, ivec & upper, vec & trunpt, mat & M, int & dim){

double sum, u, p;

vec z = zeros(dim);
vec bounds = zeros(2);

double from, to, res, count;

for(int i = 0; i < dim; i++){

sum = mu(i);
count = 0.0;

from = i * (i + 1) / 2;
to = from + i + 1;

for(int j = from; j < to-1; j++){

res = L_vec[j] * z[count]; 
sum += res;
M(i,k) += res;
count += 1;

}

bounds(upper(i)) = normalCDF((trunpt(i) - sum) / L_vec[to-1]);
bounds(1 - upper(i)) = double(1 - upper(i));

u = runif_cpp();

p = u*bounds(0) + (1-u)*bounds(1);

while(p < 0.000001 || p > 0.99999){

u = runif_cpp();

p = u*bounds(0) + (1-u)*bounds(1);

}

z(i) = NormalCDFInverse(p);

M(i,k) += L_vec[to-1] * z[count]; 

}

}



// [[Rcpp::export]] 
mat OneSideTruncMvnrv(vec & mu, vec & L_vec, ivec & above, vec & trunpt, int & n, int & ncores){

int dim = above.n_elem;

mat M = zeros(dim, n);

omp_set_num_threads(ncores);
#pragma omp parallel for
for(int k = 0; k < n; k++){TruncMvnrv_oneside_single(mu, L_vec, k, above, trunpt, M, dim);};

return M;

}



// [[Rcpp::export]]
double ghk_two_side_single(vec & L_vec, vec & above, vec & below, int & dim){

double mu, mu2, prod, prod2, x, u, Fa, Fb;
int from, to, count;

vec z = zeros(dim);
vec z2 = zeros(dim);

prod = 1.0;
prod2 = 1.0;

for(int i = 0; i < dim; i++){

from = i * (i + 1) / 2;
to = from + i + 1;

mu = 0.0;
mu2 = 0.0;

count = 0;

for(int j = from; j < to-1; j++){

mu += L_vec(j) * z(count);
mu2 += L_vec(j) * z2(count);

count += 1;

}

u = runif_cpp();

Fa = normalCDF((above[i] - mu) / L_vec[to-1]);
Fb = normalCDF((below[i] - mu) / L_vec[to-1]);

x = (Fb - Fa)*u + Fa;

if(x > 0.9999999){

	x = 0.9999999;

}

if(x < 0.0000001){

	x = 0.0000001;

}

z(i) += NormalCDFInverse(x);

prod *= Fb - Fa;

Fa = normalCDF((above[i] - mu2) / L_vec[to-1]);
Fb = normalCDF((below[i] - mu2) / L_vec[to-1]);

x = (Fb - Fa)*(1-u) + Fa;

if(x > 0.9999999){

	x = 0.9999999;

}

if(x < 0.0000001){

	x = 0.0000001;

}

z2(i) += NormalCDFInverse(x);

prod2 *= Fb - Fa;

}

return (prod + prod2)/2;
	
}



// [[Rcpp::export]]
double ghk_two_side(vec & L_vec, vec & above, vec & below, int & n, int & ncores){

double sum = 0.0;
int dim = above.n_elem;

if(n > 1){n = int(n/2);}

omp_set_num_threads(ncores);
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < n; i++){
    sum += ghk_two_side_single(L_vec, above, below, dim);
}

return sum / n;

}



// [[Rcpp::export]]
double ghk_oneside_single(vec & L_vec, vec & trunpt, ivec & is_upper, int & dim){

double mu, mu2, prod, prod2, x, x2, u;
int from, to, count;

vec z = zeros(dim);
vec z2 = zeros(dim);
vec bounds(2);
vec bounds2(2);

prod = 1.0;
prod2 = 1.0;

for(int i = 0; i < dim; i++){

from = i * (i + 1) / 2;
to = from + i + 1;

mu = 0.0;
mu2 = 0.0;
count = 0;

for(int j = from; j < to-1; j++){

mu += L_vec(j) * z(count);
mu2 += L_vec(j) * z2(count);
count += 1;

}

u = runif_cpp();

bounds(is_upper(i)) = normalCDF((trunpt(i) - mu) / L_vec(to-1));
bounds(1 - is_upper(i)) = double(1 - is_upper(i));

bounds2(is_upper(i)) = normalCDF((trunpt(i) - mu2) / L_vec(to-1));
bounds2(1 - is_upper(i)) = double(1 - is_upper(i));

x = (bounds(1) - bounds(0))*u + bounds(0);
x2 = (bounds2(1) - bounds2(0))*(1-u) + bounds2(0);

if(x > 0.9999999){

	x = 0.9999999;

}

if(x < 0.0000001){

	x = 0.0000001;

}


if(x2 > 0.9999999){

	x2 = 0.9999999;

}

if(x2 < 0.0000001){

	x2 = 0.0000001;

}

z(i) += NormalCDFInverse(x);
z2(i) += NormalCDFInverse(x2);

prod *= bounds(1) - bounds(0);
prod2 *= bounds2(1) - bounds2(0);

}

return (prod + prod2) / 2;

}




// [[Rcpp::export]]
double ghk_oneside(vec & L_vec, vec & trunpt, ivec & is_upper, int & n, int & ncores){

double sum = 0.0;
int dim = is_upper.n_elem;

if(n > 1){n = int(n/2);}

omp_set_num_threads(ncores);
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < n; i++){
    sum += ghk_oneside_single(L_vec, trunpt, is_upper, dim);
}

return sum / n;

}



