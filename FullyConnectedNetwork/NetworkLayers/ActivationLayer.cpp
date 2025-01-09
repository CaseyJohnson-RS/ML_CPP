#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER

#include <cmath>
#include "Matrix.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/*
    #### Класс слоя активации

    Во-первых, слою активации необязательно запоминать вообще
    что-либо, но этот класс запоминает тип активационной
    функции. Поэтому, при вызове Forward и Backward будут вызваны
    нужные функции.

    Во-вторых, как это работает? 
    
    ##### Forward - прямое прохождение

    Тут всё просто. Ко входному вертикальному вектору к каждому 
    элементу применяется определённая функция активации

    ##### Backward - обратное распространение

    Тут чуть сложнее. Необходимо знать как выглядит формула 
    сложной производной:

    [ f(g(x)) ]' = f'(g(x)) * g\'(x)

    В нашем случае получается так:

    f_next_layer( activation( f_previous_layer(x) ) )

    Если мы возьмём производную, то мы получим

    f_next_layer'( activation( f_previous_layer(x) ) ) * activation'( f_previous_layer(x) ) * f_previous_layer'(x)
    
    Обозначим

    grad2 = f_next_layer'( activation( f_previous_layer(x) ) )
    act_local_grad = activation'( f_previous_layer(x) )
    prev_local_grad = f_previous_layer'(x)

    Здесь:
     - grad2 - Градиент, который был посчитан, проходя через все слои спереди
     - act_local_grad - локальная производная для активационного слоя
     - prev_local_grad - локальная производная для предыдущего слоя
    
    Наша задача посчитать grad1, чтобы предыдущий слой смог посчитать grad0:

    grad1 = act_local_grad * grad2

*/
class ActivationLayer
{
public:
    enum ActivationType { None, Sigmoid, Tanh, ReLU, LeakyReLU, SoftPlus};

    ActivationLayer()
    {
        activation_function = identical;
        d_activation_function = d_identical;
    }

    ActivationLayer(ActivationType activation_type)
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

    Matrix Forward(Matrix X)
    {
        if (X.getW() != 1)
        {
            std::cout << "Input X must be vertical vector!" << std::endl;
            throw;
        }

        Matrix Y = Matrix(X.getH(), 1);

        for(unsigned int input_index = 0; input_index < X.getH(); ++input_index)
            Y[input_index][0] = (this->*activation_function)(X[input_index][0]);

        return Y;
    }

    Matrix Backward(Matrix X, Matrix gradFromNextLayer)
    {
        if (X.getW() != 1)
        {
            std::cout << "Input X must be vertical vector!"  << std::endl;
            throw;
        }

        if (gradFromNextLayer.getW() != 1)
        {
            std::cout << "Gradient from next layer must be vertical vector!" << std::endl;
            throw;
        }

        Matrix gradFromCurrLayer = Matrix(X.getH(), 1);

        for(unsigned int input_index = 0; input_index < X.getH(); ++input_index)
        {
            gradFromCurrLayer[input_index][0] = (this->*d_activation_function)(X[input_index][0]) * gradFromNextLayer[input_index][0];
        }

        return gradFromCurrLayer;
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