
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/storage/storage.h"
#include "taco/storage/pack.h"
#include "taco/util/strings.h"
using namespace taco;

#include <iostream>
using namespace std;

// The sigmoid activation function
class Sigmoid
{
private:
    //typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	//typedef Tensor<double> Z;
	
public:
    // a = activation(z) = 1 / (1 + exp(-z))
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static inline void activate(const Tensor<double> & Z, Tensor<double> & A)
    {
		IndexVar i, j;
        A(i,j) = Z(i,j);
		A.evaluate();
		//A.compile();
		//A.assemble();
		//A.compute();
    }

	
};


int main()
{
	Format csr({Dense,Dense});
	Tensor<double> A({3,3},csr);
	Tensor<double> Z({3,3},csr);
	TensorVar as("a", Float64);
	as = 1.0;
	//TensorData<double> as({1},Float64);
	 
	Z.insert({0,0}, 1.0);
	Z.insert({1,2}, 2.0);

	//Sigmoid * mySig = new Sigmoid();
	//mySig->activate(Z,A);
	
	//IndexVar i, j;
    //A(i,j) = Z(i,j)*as;
	//A.evaluate();
	cout << "as = " << as.second << endl;
	
	cout << Z << endl;
	cout << "======================\n";
	//cout << A << endl;
	//A.compile();
	//A.assemble();
	//A.compute();
	
	
	return 0;
}
	