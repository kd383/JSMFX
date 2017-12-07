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
    A.Rectification_Full();
    mat C = A.C();
    C.save("rect.csv",arma::csv_ascii);
    /*	
    A.CalculateColSum();
    A.FindAnchor();
    arma::uvec S = A.Anchor();
    cout << S << endl;
    START = omp_get_wtime();
    A.FindB();
    END = omp_get_wtime();
    cout << "ADMM takes " << END-START << " [s]" << endl;
    A.FindA();
    */
}
