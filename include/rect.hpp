#ifndef _RECTI_HPP_
#define _RECTI_HPP_
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

using arma::mat;
using arma::sp_mat;
using arma::vec;
using arma::rowvec;
using namespace arma::newarp;

const int PGD_T = 100;
const float PGD_GAMMA = 1.0;

class RectOp {
    public:
    mat* X_;
    sp_mat corNN_;
    double corJS_;
    int n_rows;
    void SetCorNN()
    {
        mat XXt = (*X_)*(*X_).t();
        corNN_ = sp_mat(-arma::clamp(XXt,XXt.min(),0));   
    }
    RectOp(mat* X, int n) 
    { 
        X_ = X;
        n_rows = n;
        SetCorNN();
        corJS_ = (1 - arma::accu(corJS_) - arma::norm(arma::square(arma::sum((*X_)))))/(n_rows*n_rows);
    }
    int rows() { return n_rows; }
    int cols() { return n_rows; }
    void perform_op (double *x_in, double *y_out) const
    {
        vec y(y_out,n_rows,false,true);
        vec x(x_in,n_rows,false,true);
        y = (*X_)*((*X_).t()*x) + corNN_*x + corJS_*arma::sum(x);
    }
};
 
void eigs(const mat C, const int k, vec& e, mat& V)
{
    DenseGenMatProd<double> op(C);
    SymEigsSolver< double, EigsSelect::LARGEST_ALGE, DenseGenMatProd<double> >eigs(op, k, 2*k);

    eigs.init();
    int nconv = eigs.compute();
    if (nconv > 0) {
        e = eigs.eigenvalues();
        V = eigs.eigenvectors();
    } else {
        std::cout << "Did not converge" << std::endl;
        return; 
    }
}

void eigs(RectOp& op, const int k, vec& e, mat& V)
{
    SymEigsSolver< double, EigsSelect::LARGEST_ALGE, RectOp >eigs(op, k, 2*k);

    eigs.init();
    int nconv = eigs.compute();
    if (nconv > 0) {
        e = eigs.eigenvalues();
        V = eigs.eigenvectors();
    } else {
        std::cout << "Did not converge" << std::endl;
        return; 
    }
}

void nearestPSD(mat& C, const int k)
{
    vec e;
    mat V;
    eigs(C, k, e, V);
    e = arma::clamp(e, 0, arma::max(e));
    C = V*arma::diagmat(e)*V.t();
}

void nearestJS( mat& C, const int n )
{
    C += (1-arma::accu(C)) / (n * n);
}

void nearestNN( mat& C )
{
    C = arma::clamp(C, 0.0, C.max());
}

void ProjGradDescent(const mat C, const int k, mat& X)
{
    vec e;
    mat V;
    eigs(C, k, e, V);
    arma::rowvec s = 2*arma::conv_to<rowvec>::from(arma::sum(arma::square(arma::clamp(V,V.min(),0)))<0.5)-1;
    V.each_row() %= s;
    X = V*arma::diagmat(arma::sqrt(e));
    mat XtX;
    double c;

    for (int i = 0; i < PGD_T; i++) {
        XtX = X.t()*X;
        c = PGD_GAMMA * arma::norm(XtX,"fro");
        X -= (X*XtX - C*X)/c;
        X = arma::clamp(X, 0, X.max());      
    }
}

template <unsigned int N, typename std::enable_if <N >= 100> :: type* = nullptr> 
void my_function()
{
    std::cout << "N >= 100" << std::endl;
}

template <unsigned int N, typename std::enable_if <N < 100> :: type* = nullptr> 
void my_function()
{
   std::cout << "N < 100" << std::endl;
}

#endif
