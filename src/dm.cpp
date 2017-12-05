// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <math.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;

const double log2pi = std::log(2.0 * M_PI);

//' @export
// [[Rcpp::export]]
double ddm(arma::vec x, arma::vec alpha){
  double left = R::gammafn( arma::sum(alpha) ) / R::gammafn( arma::sum(x+alpha) );
  double right = 1;
  for(int i=0;i<x.size();i++){
    right *= R::gammafn(x[i] + alpha[i])/R::gammafn(alpha[i]);
  }
  return( left * right );
}


arma::vec dm_iter(arma::mat X, arma::vec alpha){
  int K = X.n_cols;
  int n = X.n_rows;
  double alpha_total = 0;
  for(int j=0; j<K; j++){
    alpha_total += alpha(j);
  }
  double psi_alpha_total = R::digamma(alpha_total);
  arma::vec N = arma::vec(n);
  for(int i=0; i<n; i++){
    N(i) = 0;
    for(int j=0; j<K; j++){
      N(i) += X(i,j);
    }
  }
  for(int j=0; j<K; j++){
    double numerator = 0, denominator = 0;
    for(int i=0; i<n; i++){
      numerator += (R::digamma(X(i,j) + alpha(j)) - R::digamma(alpha(j)));
      denominator += (R::digamma(N(i) + alpha_total) - psi_alpha_total);
    }
    alpha[j] = alpha[j] * numerator / denominator;
  }
  return(alpha);
}

//' @export
// [[Rcpp::export]]
Rcpp::List c_dm_fit(arma::mat X, double eps = 0.0001, int maxiter = 5000){
  int K = X.n_cols;
  arma::vec alpha_prev = arma::ones<arma::vec>(K);
  arma::vec alpha;
  int iter = 0;
  double err = eps + 1;
  while(err > eps & iter < maxiter){
    alpha = dm_iter(X, alpha_prev);
    err = arma::norm(alpha - alpha_prev);
    alpha_prev = alpha;
    iter++;
  }
  return Rcpp::List::create(alpha, iter);
}

//' @export
// [[Rcpp::export]]
arma::vec c_dm_fit_alpha(arma::mat X, double eps = 0.0001, int maxiter = 5000){
  int K = X.n_cols;
  arma::vec alpha_prev = arma::ones<arma::vec>(K);
  arma::vec alpha;
  int iter = 0;
  double err = eps + 1;
  while(err > eps & iter < maxiter){
    alpha = dm_iter(X, alpha_prev);
    err = arma::norm(alpha - alpha_prev);
    alpha_prev = alpha;
    iter++;
  }
  return alpha;
}
