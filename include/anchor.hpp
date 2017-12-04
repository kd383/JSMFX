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


	public:
		anchor(const mat X) { X_ = mat(X); n_ = X_.n_rows; k_ = X_.n_cols;}
		anchor(const char* str) { X_.load(str); n_ = X_.n_rows; k_ = X_.n_cols;}
		const mat& X() const { return X_; }
		const colvec& ColSum() const { return colSum_; }
		void CalculateColSum();
		const mat& R() const { return R_; }
		void CalculateR();
		const uvec Anchor() const { return anchor_; }
		void FindAnchor();
		const mat& B() const { return B_; }
		void FindB();
		const mat& A() const { return A_; }
		void FindA();
};

void anchor::CalculateColSum()
{
	colvec xsum = conv_to<colvec>::from(sum(X_,0));
	colSum_ = X_*xsum;
	return;
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
	RXbar_ = X_.each_col()/colSum_;
	RXbar_ = R_*RXbar_.t();
	vector<int> piv(n_); 
	QRCP(RXbar_, piv);
	piv.resize(k_);
	anchor_ = conv_to<uvec>::from(piv) - 1;

	return;
}

void anchor::FindB()
{
	mat H = RXbar_.cols(anchor_);
	mat HtH = H.t()*RXbar_;
	mat HtHS = HtH.cols(anchor_);
	mat y0 = arma::solve(HtHS,HtH);
	mat F = arma::inv(GAMMA*HtHS + arma::eye(arma::size(HtHS)));
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
	mat XS = X_.rows(anchor_);
	A_ = XS*XS.t();
	vec denom = arma::diagvec(B_.rows(anchor_));
	A_.each_col() /= denom;
	A_.each_row() /= denom.t();
}
#endif