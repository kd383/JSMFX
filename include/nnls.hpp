#ifndef _NNLS_HPP_
#define _NNLS_HPP_
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

using arma::vec;
using arma::mat;
using std::cout;
using std::endl;

const float TOL = 1e-5;
const int MAXIT = 500;
const float LAMBDA = 1.9;

vec ProjectToSimplex(const vec y)
{
	vec u = arma::sort(y,"descend");
	vec us = 1 - arma::cumsum(u);
	vec j = arma::regspace<vec>(1, y.n_elem);
	vec c = u + (1/j)%us;
	arma::uvec rho = arma::find(c > 0, 1, "last");
	vec lambda = (1/(1+(double)arma::as_scalar(rho)))*us(rho);
	vec x = arma::clamp(y + arma::as_scalar(lambda), 0 ,y.max() + arma::as_scalar(lambda));
	return x;
}

vec ADMM_DR(const mat F, const vec f, const vec y0)
{
	vec b = y0, y = y0, prev_y, a;
	for (int t = 0; t < MAXIT; t++) {
		prev_y = y;
		a = F*(2*y-b+f);
		b += LAMBDA*(a-y);
		y = ProjectToSimplex(b);
		if (arma::norm(y-prev_y,2) < TOL)
			break;
	}
	return y;
}

#endif
