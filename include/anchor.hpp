#ifndef _ANCHOR_HPP_
#define _ANCHOR_HPP_
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <vector>
#include "lapack.hpp"
#include "nnls.hpp"
#include "rect.hpp"

using std::string;
using arma::mat;
using arma::colvec;
using arma::rowvec;
using arma::uvec;
using arma::ivec;
using arma::conv_to;
using std::vector;

const float GAMMA = 3.0;
const int RECT_T = 50;

class anchor{
    private:

    int n_, k_;
    bool isLowRank_;

    // Low rank variable
    mat X_;
    mat R_;
    mat RXbar_;

    // Full rank variable
    mat Craw_;
    mat C_;
    mat Cbar_;

    // Anchor word output
    colvec colSum_;
    uvec anchor_;
    mat B_;
    mat A_;


    public:

    // Initialization
    void init(const mat C);
    anchor(const mat C) { init(C); }
    anchor(const mat C, const int k) { init(C); k_ = k; }
    anchor(const char* str);
    anchor(const char* str, const int k): anchor(str) { k_ = k;}

    // Getter
    const mat& C() const { return C_; }
    const mat& X() const { return X_; }
    const colvec& ColSum() const { return colSum_; }
    const mat& R() const { return R_; }
    const uvec Anchor() const { return anchor_; }
    const mat& B() const { return B_; }
    const mat& A() const { return A_; }

    // Anchor word algorithm
    void CalculateColSum();
    void CalculateR();
    void FindAnchor();
    void FindB();
    void FindA();

    // Rectification
    void Rectification_Full();
    void Rectification_LowRank();
    void CalculateX();
};

void anchor::init(const mat C)
{
    if ( C.n_rows == C.n_cols) {
        C_ = mat(C);
        n_ = C.n_rows;
        k_ = 20;
        isLowRank_ = false;
    } else {
        X_ = mat(C);
        n_ = C.n_rows;
        k_ = C.n_cols;
        isLowRank_ = true;
    }
}

anchor::anchor(const char* str)
{
    mat C;
    C.load(str);
    init(C);
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
            B.col(i) = ADMM_DR(F, f, ProjectToSimplex(y0.col(i)));
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

void anchor::Rectification_Full()
{
    Craw_ = C_;
    for (int i = 0; i < RECT_T; i++) {
        nearestPSD(C_, k_);
        nearestJS(C_, n_);
        nearestNN(C_);
    }
    C_ /= arma::accu(C_);
}

void anchor::Rectification_LowRank()
{
    vec e;
    mat V;
    eigs(C_, k_, e, V);
    X_ = V*arma::diagmat(arma::sqrt(e));
    for (int i = 0; i < RECT_T; i++) {
        RectOp op(&X_, n_);
        eigs(op, k_, e, V); 
        X_ = V*arma::diagmat(arma::sqrt(e));
    }
    
    X_ /= arma::norm(arma::sum(X_),"fro");
    isLowRank_ = true;
}

void anchor::CalculateX()
{
    if (Craw_.is_empty())
        ProjGradDescent(C_, k_, X_);
    else
        ProjGradDescent(Craw_, k_, X_);
    X_ /= arma::norm(arma::sum(X_),"fro");

    isLowRank_ = true;
}
#endif
