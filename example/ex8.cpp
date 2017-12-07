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
    anchor A("../dataset/nips_full.bin");
    A.Rectification_LowRank();
    A.X().save("X.csv",arma::csv_ascii);
}
