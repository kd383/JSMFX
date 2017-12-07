#include <armadillo>
using namespace arma::newarp;
using arma::mat;
using arma::vec;

class myOP {
    public:
    mat* C_;
    int n_rows;
    myOP(mat* C, int n) { C_=C; n_rows = n;}
    int rows() { return n_rows; }
    int cols() { return n_rows; }
    void perform_op (double *x_in, double *y_out) const
    {
        vec y(y_out,n_rows,false,true);
        vec x(x_in,n_rows,false,true);
        y = (*C_)*x; 
    } 
};





int main()
{
    // We are going to calculate the eigenvalues of M
    arma::mat C;
    C.load("../dataset/nips_full.bin");
    //myOP op(&C,5000);
    // Construct matrix operation object using the wrapper class DenseGenMatProd
    DenseGenMatProd<double> op(C);
    std::cout << arma::size(C) << std::endl;
    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver< double, EigsSelect::LARGEST_ALGE, DenseGenMatProd<double> > eigs(op, 20, 40);
    //SymEigsSolver< double, EigsSelect::LARGEST_ALGE, myOP > eigs(op, 20, 40);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    arma::vec evalues;
    if(nconv > 0)
     evalues = eigs.eigenvalues();

    evalues.print("Eigenvalues found:");
    /*
    arma::mat evector;
    if(nconv > 0)
     evector = eigs.eigenvectors();

    evector.print("Eigenvectors found:");
    */
    return 0;
}
