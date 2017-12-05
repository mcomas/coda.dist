// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "coda.h"
#include "nm.h"
#include "dm.h"
#include "hermite.h"
#include <random>
#include <vector>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

/*
*
* - expected_hermite: exact method using hermite poylinomials
* - expected_montecarlo_01: montecarlo approximation centered at mu_exp. sigma_ilr used as sampling variance.
* - expected_montecarlo_02: montecarlo approximation centered at mu_exp. sigma_ilr with total variance sigma_exp as sampling variance.
* - expected_montecarlo_03: metropolis sampling
* - expected_metropolis: montecarlo approximation centered at mu_exp. sigma_exp as sampling variance.
*/

using namespace Rcpp;

//' @export
// [[Rcpp::export]]
double lpnm_join_no_constant(arma::vec x, arma::vec mu, arma::mat inv_sigma,
                             arma::vec p, arma::vec h){
  double lmult = arma::accu(x % log(p));
  arma::vec y = h-mu;
  double lnormal = -0.5 * ((arma::mat)(y.t() * inv_sigma * y))(0,0);
  return(lmult + lnormal);
}

//' @export
// [[Rcpp::export]]
Rcpp::List expected_hermite(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, int order){
  double M0 = dnm(x, mu_ilr, sigma_ilr, order);
  arma::vec M1 = m1_dnm(x, mu_ilr, sigma_ilr, order)/M0;
  arma::mat M2 = m2_dnm(x, mu_ilr, sigma_ilr, order)/M0;
  return(Rcpp::List::create(M0, M1, M2));
}

//' @export
// [[Rcpp::export]]
Rcpp::List expected_montecarlo_01(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, arma::mat Z,
                      arma::vec mu_exp){ //, double var_exp){

  int K = x.size();
  int k = K - 1;
  int nsim = Z.n_rows;

  arma::mat inv_sigma = inv_sympd(sigma_ilr);

  arma::vec sampling_mu =  mu_exp;
  arma::mat SAMPLING_MU = arma::repmat(sampling_mu.t(), nsim, 1);

  arma::mat sampling_sigma = sigma_ilr;
  arma::mat sampling_inv_sigma = inv_sigma;
  arma::mat sampling_sigma_chol = arma::chol(sampling_sigma);

  //arma::mat mu12 = 0.5 * (sampling_mu.t() * inv_sigma * sampling_mu - mu_ilr.t() * inv_sigma * mu_ilr);
  arma::mat D = inv_sigma * (mu_ilr-sampling_mu);

  //double mult_const = lpmultinomial_const(x);

  double M0 = 0;
  arma::vec M1 = arma::vec(k);
  arma::mat M2 = arma::mat(k,k);
  M1.zeros();M2.zeros();
  arma::mat Hs = SAMPLING_MU + Z * sampling_sigma_chol;
  arma::mat Ps = inv_ilr_coordinates(Hs);
  arma::vec loglik = lpmultinomial_mult(Ps, x) + Hs * D; //mu12(0,0) +
  double cmax = max(loglik);

  arma::vec lik = exp(loglik - cmax);
  arma::vec lik_st = lik / mean(lik);

  M0 += mean(lik);
  for(int i = 0;i < k; i++){
    arma::vec C = Hs.col(i) % lik_st;
    M1(i) += mean(C);
    for(int j = 0;j < k; j++){
      M2(i,j) += mean(C % Hs.col(j));
    }
  }
  return(Rcpp::List::create(M0, M1, M2));
}

//' @export
// [[Rcpp::export]]
arma::mat expected_montecarlo_02(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, arma::mat Z,
                     arma::vec mu_exp, double var_exp){

  int K = x.size();
  int k = K - 1;
  int nsim = Z.n_rows;

  arma::mat inv_sigma = inv_sympd(sigma_ilr);

  arma::vec sampling_mu =  mu_exp;
  arma::mat SAMPLING_MU = arma::repmat(sampling_mu.t(), nsim, 1);

  arma::mat sampling_sigma = var_exp * sigma_ilr;
  arma::mat sampling_inv_sigma = inv_sigma / var_exp;
  arma::mat sampling_sigma_chol = arma::chol(sampling_sigma);

  //arma::mat mu12 = 0.5 * (sampling_mu.t() * inv_sigma * sampling_mu - mu_ilr.t() * inv_sigma * mu_ilr);
  arma::mat D = inv_sigma * (mu_ilr-sampling_mu);

  //double mult_const = lpmultinomial_const(x);


  arma::mat M = arma::mat(1+k,k);
  M.zeros();
  arma::mat Hs = SAMPLING_MU + Z * sampling_sigma_chol;
  arma::mat Ps = inv_ilr_coordinates(Hs);
  arma::vec loglik = lpmultinomial_mult(Ps, x) +
    0.5 * sum(((1/var_exp-1) * Hs * inv_sigma) % Hs, 1) +
    Hs * inv_sigma * (mu_ilr - 1/var_exp * sampling_mu);
  double cmax = max(loglik);

  arma::vec lik = exp(loglik - cmax);
  arma::vec lik_st = lik / mean(lik);

  for(int i = 0;i < k; i++){
    arma::vec C = Hs.col(i) % lik_st;
    M(0,i) += mean(C);
    for(int j = 0;j < k; j++){
      M(1+i,j) += mean(C % Hs.col(j));
    }
  }
  return(M);
}

//' @export
// [[Rcpp::export]]
Rcpp::List expected_montecarlo_03(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, arma::mat Z,
                      arma::vec mu_sampling, arma::mat sigma_sampling){

  int K = x.size();
  int k = K - 1;
  int nsim = Z.n_rows;
  arma::mat B = ilr_basis(K);

  arma::mat inv_sigma_ilr = inv_sympd(sigma_ilr);
  arma::mat inv_sigma_sampling = inv_sympd(sigma_sampling);


  Z = Z * arma::chol(sigma_sampling);
  Z.each_row() += mu_sampling.t();
  arma::vec loglik = -ldnormal(Z, mu_sampling, inv_sigma_sampling);

  for(int i = 0; i < nsim; i++){
    arma::vec h = Z.row(i).t();
    arma::vec p = exp(B * h);
    loglik(i) += lpnm_join_no_constant(x, mu_ilr, inv_sigma_ilr, p / arma::accu(p), h);
  }
  arma::vec lik = arma::exp(loglik - max(loglik));
  arma::vec lik_st = lik / mean(lik);

  double M0 = 0;
  arma::vec M1 = arma::vec(k);
  arma::mat M2 = arma::mat(k,k);
  M1.zeros(); M2.zeros();

  M0 += mean(lik);
  for(int i = 0;i < k; i++){
    arma::vec C = Z.col(i) % lik_st;
    M1(i) += mean(C);
    for(int j = 0;j < k; j++){
      M2(i,j) += mean(C % Z.col(j));
    }
  }

  return(Rcpp::List::create(M0, M1, M2));

}

//' @export
// [[Rcpp::export]]
Rcpp::List expected_montecarlo_04(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, arma::mat Z,
                                  arma::vec m1, arma::mat m2){
  arma::vec mu_sampling = m1;
  arma::mat sigma_sampling =  m2 - m1 * m1.t();

  int K = x.size();
  int k = K - 1;
  int nsim = Z.n_rows;
  arma::mat B = ilr_basis(K);

  arma::mat inv_sigma_ilr = inv_sympd(sigma_ilr);
  arma::mat inv_sigma_sampling = inv_sympd(sigma_sampling);


  Z = Z * arma::chol(sigma_sampling);
  Z.each_row() += mu_sampling.t();
  arma::vec loglik = -ldnormal(Z, mu_sampling, inv_sigma_sampling);

  for(int i = 0; i < nsim; i++){
    arma::vec h = Z.row(i).t();
    arma::vec p = exp(B * h);
    loglik(i) += lpnm_join_no_constant(x, mu_ilr, inv_sigma_ilr, p / arma::accu(p), h);
  }
  arma::vec lik = arma::exp(loglik - max(loglik));
  arma::vec lik_st = lik / mean(lik);

  double M0 = 0;
  arma::vec M1 = arma::vec(k);
  arma::mat M2 = arma::mat(k,k);
  M1.zeros(); M2.zeros();

  M0 += mean(lik);
  for(int i = 0;i < k; i++){
    arma::vec C = Z.col(i) % lik_st;
    M1(i) += mean(C);
    for(int j = 0;j < k; j++){
      M2(i,j) += mean(C % Z.col(j));
    }
  }

  return(Rcpp::List::create(M0, M1, M2));

}

//' @export
// [[Rcpp::export]]
Rcpp::List expected_metropolis(arma::vec x, arma::vec mu_ilr, arma::mat sigma_ilr, arma::vec mu_exp,
                               int nsim, int ignored_steps = 100){
  int maxZ = 10000;
  int maxU = 10000;

  int K = x.size();
  int k = K - 1;

  arma::mat Z = arma::randn(k, maxZ);
  arma::vec U = arma::randu(maxU);

  arma::mat inv_sigma = inv_sympd(sigma_ilr);
  arma::mat B = ilr_basis(K);

  // initialisation
  arma::vec h = mu_exp, h_proposal;
  arma::vec p = exp(B * h), p_proposal;

  double f = lpnm_join_no_constant(x, mu_ilr, inv_sigma, p / arma::accu(p), h);

  int irand_z = 0, irand_u = 0, step = 0, nsim_real = 0;
  bool repeat = true;
  double M0 = 0;
  arma::vec M1 = arma::zeros(k);
  arma::mat M2 = arma::mat(k,k);

  for(int i = 0; i < nsim; i++){
    for(int j = 0; j < ignored_steps; j++){
      h_proposal = h + Z.col(irand_z++);
      p_proposal = exp(B * h_proposal);
      double f_proposal = lpnm_join_no_constant(x, mu_ilr, inv_sigma,
                                                p_proposal / arma::accu(p_proposal), h_proposal);

      double cmean = 0.5 * f_proposal + 0.5 * f;
      double alpha = exp(f_proposal-cmean) / exp(f-cmean);
      if(1 < alpha){
        f = f_proposal;
        h = h_proposal;
      }else{
        if(U(irand_u++) < alpha){
          f = f_proposal;
          h = h_proposal;
        }else{
          f = f;
          h = h;
        }
      }
      if(irand_z == maxZ){
        Z = arma::randn(k, maxZ);
        irand_z = 0;
      }
      if(irand_u == maxU){
        U = arma::randu(maxU);
        irand_u = 0;
      }
    }

    M0 += f;
    M1 += h;
    M2 += h * h.t();
  }
  M0 /= nsim;
  M1 /= nsim;
  M2 /= nsim;

  return(Rcpp::List::create(M0, M1, M2));
}


