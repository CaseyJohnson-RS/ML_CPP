#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER

#include <cmath>
#include "../Tensor.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ActivationLayer
{

public:

    enum ActivationType { None, Sigmoid, Tanh, ReLU, LeakyReLU, SoftPlus };

    ActivationLayer()
    {
        activation_function = identical;
        d_activation_function = d_identical;
    }

    void SetActivationType(ActivationType activation_type)
    {        
        switch (activation_type)
        {
        case ActivationType::Sigmoid:
            activation_function = sigmoid;
            d_activation_function = d_sigmoid;
            break;
        case ActivationType::LeakyReLU:
            activation_function = leaky_relu;
            d_activation_function = d_leaky_relu;
            break;
        case ActivationType::ReLU:
            activation_function = relu;
            d_activation_function = d_relu;
            break;
        case ActivationType::SoftPlus:
            activation_function = softplus;
            d_activation_function = d_softplus;
            break;
        case ActivationType::Tanh:
            activation_function = tanh;
            d_activation_function = d_tanh;
            break;
        case ActivationType::None:
            activation_function = identical;
            d_activation_function = d_identical;
            break;
        }
    }

    Tensor Forward(const Tensor& X)
    {
        Tensor Y = Tensor(X.get_depth(), X.get_height(), X.get_width());

        for(unsigned int d = 0; d < X.get_depth(); ++d)
        {
            for(unsigned int h = 0; h < X.get_height(); ++h)
            {
                for(unsigned int w = 0; w < X.get_width(); ++w)
                {
                    Y[d][h][w] = (this->*activation_function)(X(d, h, w));
                }
            }
        }

        return Y;
    }

    Tensor Backward(const Tensor& X, const Tensor& GFNL)
    {
        Tensor GFCL = Tensor(X.get_depth(), X.get_height(), X.get_width());

        for(unsigned int d = 0; d < X.get_depth(); ++d)
        {
            for(unsigned int h = 0; h < X.get_height(); ++h)
            {
                for(unsigned int w = 0; w < X.get_width(); ++w)
                {
                    GFCL[d][h][w] = (this->*d_activation_function)(X(d, h, w)) * GFNL(d, h, w);
                }
            }
        }

        return GFCL;
    }
    
private:

    double (ActivationLayer::*activation_function) (double);
    double (ActivationLayer::*d_activation_function) (double);

    double identical(double x)
    {
        return x;
    }

    double d_identical(double x)
    {
        return 1;
    }


    double sigmoid(double x) 
    { 
        return 1/(1 + exp(-x)); 
    }

    double d_sigmoid(double x) 
    {
        double sigm = sigmoid(x);
        return sigm * (1 - sigm);
    }


    double tanh(double x)
    {
        return std::tanh(x);
    }

    double d_tanh(double x)
    {
        return 1 - pow(std::tanh(x),2);
    }


    double relu(double x)
    {
        return x >= 0 ? x : 0;
    }

    double d_relu(double x)
    {
        return x >= 0 ? 1 : 0;
    }


    double leaky_relu(double x)
    {
        return x >= 0 ? x : 0.01 * x;
    }

    double d_leaky_relu(double x)
    {
        return x >= 0 ? 1 : 0.01;
    }


    double softplus(double x)
    {
        return log(1 + exp(x));
    }

    double d_softplus(double x)
    {
        return sigmoid(x);
    }

};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif