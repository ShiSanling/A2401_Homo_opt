#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include<fstream>
#include<iostream>
#include <Eigen/Dense>
#include<Eigen/Sparse>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> T;

MatrixXd add_mat(MatrixXd A_mat, MatrixXd B_mat)
{
  return A_mat.inverse();
}
Eigen::MatrixXf linalg_solve(SpMat A_mat,SpMat B_mat)
{
<<<<<<< HEAD
=======
	
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
>>>>>>> 7b1cbbbde5c096dc8a06e75fa343760285193ee6
	
	Eigen::ConjugateGradient<SpMat> solver(A_mat);
	return MatrixXf(solver.solve(B_mat));
}

namespace py = pybind11;

PYBIND11_MODULE(linalg_solve_moudle, m)

{

    m.doc() = "linalg solver";//解释说明

    m.def("linalg_solve"/*在pyhon中使用的函数名*/, &linalg_solve);

}
