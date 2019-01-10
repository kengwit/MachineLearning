#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/storage/storage.h"
#include "taco/storage/pack.h"
#include "taco/util/strings.h"
using namespace taco;

// The sigmoid activation function
class Sigmoid
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Tensor<double> Z;
	
public:
    // a = activation(z) = 1 / (1 + exp(-z))
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static inline void activate(const Tensor<double> & Z, Tensor<double> & A)
    {
		IndexVar i;
        A(i) = 1.0 / ( 1.0 + Z(i) );
		
		A.compile();
		A.assemble();
		A.compute();
    }

	// Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(a .* (1 - a))
    // g = J * f = a .* (1 - a) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    //static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
    //{
    //    G.array() = A.array() * (Scalar(1) - A.array()) * F.array();
    //}
};



#endif /* ACTIVATION_SIGMOID_H_ */
