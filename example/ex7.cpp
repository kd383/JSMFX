#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include "anchor.hpp"
#include "lapack.hpp"
#include "omp.h"

using arma::mat;
using arma::randn;
using std::cout;
using std::endl;

int main() {
    double START,END;
    mat X(6,4);
    X.randn();
    cout << X*X.t() << endl;
    RectOp ro(&X,6);
    ro.corNN_.print();
    vec x(6), y(6);
    x.randn(6);
    ro.perform_op(x.memptr(),y.memptr());
    y.print();
}
