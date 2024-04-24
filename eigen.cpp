#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include<fstream>
#include<iostream>
#include <Eigen/Dense>
#include<Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#define EIGEN_USE_THREADS
using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> T;

MatrixXd add_mat(MatrixXd A_mat, MatrixXd B_mat)
{
  return A_mat.inverse();
}
// Eigen::MatrixXf linalg_solve(SpMat A_mat,SpMat B_mat)
// {
// 	Eigen::initParallel();
// 	Eigen::BiCGSTAB<SpMat,Eigen::IncompleteLUT<float>> solver(A_mat);

//   solver.setMaxIterations(1000);
//   solver.setTolerance(1e-6);
//   Eigen::MatrixXf x = solver.solve(MatrixXf(B_mat));
//   std::cout << "#iterations: " << solver.iterations() << std::endl;
//   std::cout << "Estimated error: " << solver.error() << std::endl;
// 	return x;
// }
Eigen::MatrixXf linalg_solve(SpMat A_mat,SpMat B_mat)
{
	
	Eigen::ConjugateGradient<SpMat> solver(A_mat);
  solver.setTolerance(1e-6);
	return MatrixXf(solver.solve(B_mat));
}
// Eigen::MatrixXf linalg_solve(SpMat A_mat,SpMat B_mat)
// {
// 	Eigen::initParallel();
// 	Eigen::ConjugateGradient<SpMat> solver(A_mat);
// //   solver.setMaxIterations(1000);
//   solver.setTolerance(1e-6);
//   Eigen::MatrixXf x = solver.solve(MatrixXf(B_mat));
// 	return x;
// }



Eigen::MatrixXf linalg_inverse(SpMat A_mat)
{
	
  Eigen::ConjugateGradient<SpMat> solver(A_mat);
	return MatrixXf(solver.solve(Eigen::MatrixXf::Identity(A_mat.rows(),A_mat.cols())));
  // Eigen::SparseLU<SpMat> solver;
  // solver.compute(A_mat);
  // Eigen::IncompleteLUT<SpMat> lut(solver);

  // return MatrixXf(lut.inverse());

}
namespace py = pybind11;

PYBIND11_MODULE(linalg_solve_moudle, m)

{

    m.doc() = "linalg solver & inverse";//解释说明

    m.def("linalg_solve"/*在pyhon中使用的函数名*/, &linalg_solve);
    m.def("linalg_inverse"/*在pyhon中使用的函数名*/, &linalg_inverse);

}


