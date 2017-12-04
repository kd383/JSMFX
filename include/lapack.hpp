#ifndef _LAPACK_HPP_
#define _LAPACK_HPP_
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

using arma::mat;
using std::cout;
using std::endl;

extern "C" {
	void dgeqrf_(int *, int *, double *, int *, double *, double *, int *, int * );
	void dgeqp3_(int *, int *, double *, int *, int *, double *, double *, int *, int * );
}

// *********************************************************************
// QR without column pivoting. Use dgeqrf routine
// *********************************************************************

void QR( mat& A )
{
	int m = A.n_rows, n = A.n_cols, lda = A.n_rows;
	double *Aptr = A.memptr();
    int lwork=-1, info;
    double dummyWork;
    double *tau = new double[n];
    dgeqrf_( &m, &n, Aptr, &lda, tau, &dummyWork, &lwork, &info );

    lwork = dummyWork;
    std::vector<double> work(lwork);
    dgeqrf_( &m, &n, Aptr, &lda, tau, &work[0], &lwork, &info );

    return;
}

void QRCP( mat A, std::vector<int>& piv)
{
	int m = A.n_rows, n = A.n_cols, lda = A.n_rows;
	if (piv.size() != n)
		piv.resize(n);
	double *Aptr = A.memptr();
    int lwork=-1, info;
    double dummyWork;
    double *tau = new double[n];
    dgeqp3_( &m, &n, Aptr, &lda, &piv[0], tau, &dummyWork, &lwork, &info );

    lwork = dummyWork;
    std::vector<double> work(lwork);
    dgeqp3_( &m, &n, Aptr, &lda, &piv[0], tau, &work[0], &lwork, &info );

    return;
}

#endif