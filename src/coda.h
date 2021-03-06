#include <RcppArmadillo.h>

arma::mat ilr_basis(unsigned int dim);

arma::mat ilr_coordinates(arma::mat X);

arma::mat inv_ilr_coordinates(arma::mat ilrX);

arma::mat ilr_to_alr(unsigned int dim);
