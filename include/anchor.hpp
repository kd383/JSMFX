#ifndef _ANCHOR_HPP_
#define _ANCHOR_HPP_
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <vector>
#include "lapack.hpp"
#include "nnls.hpp"

using std::string;
using arma::mat;
using arma::colvec;
using arma::rowvec;
using arma::uvec;
using arma::ivec;
using arma::conv_to;
using std::vector;

const float GAMMA = 3.0;

class anchor{
    private:
    int n_, k_;
    mat X_;
    mat R_;
    colvec colSum_;
    mat RXbar_;
    uvec anchor_;
    mat B_;
    mat A_;
    mat C_;
    mat Cbar_;
    bool isLowRank_;


    public:
    anchor(const mat X);
    anchor(const mat C, const int k):anchor(C) { k_ = k; }
    anchor(const char* str);
    anchor(const char* str, const int k):anchor(str) { k_ = k;}
    const mat& C() const { return C_; }
    const mat& X() const { return X_; }
    const colvec& ColSum() const { return colSum_; }
    const mat& R() const { return R_; }
    const uvec Anchor() const { return anchor_; }
    const mat& B() const { return B_; }
    const mat& A() const { return A_; }
    void CalculateColSum();
    void CalculateR();
    void FindAnchor();
    void FindB();
    void FindA();
};

anchor::anchor(const mat X)
{
    if ( X.n_rows == X.n_cols) {
        C_ = mat(X);
        n_ = X.n_rows;
        k_ = 20;
        isLowRank_ = false;
    } else {
        X_ = mat(X);
        n_ = X.n_rows;
        k_ = X.n_cols;
        isLowRank_ = true;
    }
}

anchor::anchor(const char* str)
{
    mat X;
    X.load(str);
    if ( X.n_rows == X.n_cols) {
        C_ = mat(X);
        n_ = X.n_rows;
        k_ = 20;
        isLowRank_ = false;
    } else {
        X_ = mat(X);
        n_ = X.n_rows;
        k_ = X.n_cols;
        isLowRank_ = true;
    }
}

void anchor::CalculateColSum()
{
    if (isLowRank_){ 
        colvec xsum = conv_to<colvec>::from(sum(X_,0));
        colSum_ = X_*xsum;
        return;
    } else {
        colSum_ = sum(C_,1);
        return;
    }
}

void anchor::CalculateR()
{
    R_ = mat(X_);
    QR(R_);
    R_ = trimatu(R_.rows(0,k_-1));
    return;
}

void anchor::FindAnchor()
{
    vector<int> piv(n_);
    if (isLowRank_) {
        RXbar_ = X_.each_col()/colSum_;
        RXbar_ = R_*RXbar_.t();
        QRCP(RXbar_, piv);
    } else {
        Cbar_ = C_.each_row()/conv_to<rowvec>::from(colSum_);
        QRCP(Cbar_, piv);
    }
    piv.resize(k_);
    anchor_ = conv_to<uvec>::from(piv) - 1;

    return;
}

void anchor::FindB()
{
    mat F, y0, HtH, H;
    if (isLowRank_) {
        H = RXbar_.cols(anchor_);
        HtH = H.t()*RXbar_;
    } else {
        mat H = Cbar_.cols(anchor_);
        HtH = H.t()*Cbar_;
    }
    mat HtHS = HtH.cols(anchor_);
    y0 = arma::solve(HtHS,HtH);
    F = arma::inv(GAMMA*HtHS + arma::eye(arma::size(HtHS)));
    vec f(k_);

    ivec ind(n_);
    ind.fill(-1);
    for (int i = 0; i < anchor_.size(); i++)
        ind[anchor_[i]] = i;

    mat B(k_, n_, arma::fill::zeros);
    for (int i = 0; i < n_; i++) {
        if (ind[i]>=0)
            B(ind[i],i) = 1;
        else {
            f = GAMMA*HtH.col(i);
            B.col(i) = ADMM_DR(F, f, y0.col(i));
        }
    }
    arma::rowvec denom = (B*colSum_).t();
    B_ = B.t();
    B_.each_col() %= colSum_;
    B_.each_row() /= denom;

    return;
}

void anchor::FindA()
{
    if (isLowRank_) {
        mat XS = X_.rows(anchor_);
        A_ = XS*XS.t();
    } else {
        A_ = C_.submat(anchor_,anchor_);
    }
    vec denom = arma::diagvec(B_.rows(anchor_));
    A_.each_col() /= denom;
    A_.each_row() /= denom.t();

   return;
}
#endif
