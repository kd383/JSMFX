#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include "omp.h"

#include "rect.hpp"

using arma::mat;
using arma::randn;
using arma::randu;
using arma::vec;
using std::cout;
using std::endl;

int main() {
    my_function<50>();
    my_function<420>();
    vec d = { 1,2,3,4};
    mat A = arma::diagmat(d);
    vec x(4); x.randn();
    x.print();
    double *y = new double[4];
    vec yv(y,4,false,true);
    yv = A*x;
    for (int i = 0; i < 4; i++) {
        cout << y[i] << endl;
    }
    delete y;

}
