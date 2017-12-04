#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include "omp.h"

#include "nnls.hpp"

using arma::mat;
using arma::randn;
using arma::randu;
using arma::vec;
using std::cout;
using std::endl;

int main() {
	arma::arma_rng::set_seed_random();
	mat A = randn<mat>(10,5);
	mat AtA = A.t()*A;
	mat F = arma::inv(3*AtA + arma::eye(5,5));
	vec c = randu<vec>(5);
	c = c/arma::sum(c);
	vec b = A*c;
	vec f = 3*A.t()*b;
	vec x0 = arma::ones<vec>(5)/5;
	cout << F << endl << f << endl << x0 << endl;
	vec x = ADMM_DR(F,f,x0);
	cout << c  << endl << x << endl;
}
