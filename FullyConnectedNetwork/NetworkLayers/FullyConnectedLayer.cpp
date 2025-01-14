#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include <iostream>
#include <string>
#include <random>
#include "Matrix.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/*
    Полносвязный слой, разобраться в котором не так сложно:

    Самая сложная часть:
    
    https://education.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki
*/
class FullyConnectedLayer
{

private:

    Matrix W;
    Matrix B;

    unsigned int inputs;
    unsigned int outputs;

    double learning_rate;

public:

    FullyConnectedLayer()
    {
        inputs = 0;
        outputs = 0;

        learning_rate = 0;
    }

    FullyConnectedLayer(unsigned int inputs, unsigned int outputs, double learning_rate = 1e-4)
    {
        this -> inputs = inputs;
        this -> outputs = outputs;
        this -> learning_rate = learning_rate;

        W = Matrix(outputs, inputs);
        B = Matrix(outputs, 1);

        DefineWeights(1,0);
    }

    /*
        Инициализация весов с помощью нормального распределения
    */
    void DefineWeights(double sigma, double mean)
    {
        // Создание генератора шума, распределенного нормально
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution d = std::normal_distribution(mean, sigma);
        auto random_double = [&d, &gen]{ return d(gen); };

        for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
        {
            B[neuron_index][0] = random_double();
            for(unsigned int input_index = 0; input_index < inputs; ++input_index)
            {
                W[neuron_index][input_index] = random_double();
            }
        }
    }

    /*
        Инициализация весов константой
    */
    void DefineWeights(double value)
    {
        for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
        {
            B[neuron_index][0] = value;
            for(unsigned int input_index = 0; input_index < inputs; ++input_index)
            {
                W[neuron_index][input_index] = value;
            }
        }
    }

    /*
        Банальное перемножение весов каждого нейрона со входным сигналом
        и суммирование
    */
    Matrix Forward(Matrix X)
    {
        if (X.getH() != inputs || X.getW() != 1)
        {
            std::cout << "Input X must be vertical vector!" << std::endl;
            throw;
        }
        
        Matrix Y = Matrix(outputs, 1);

        for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
        {
            Y[neuron_index][0] = B[neuron_index][0];

            for(unsigned int input_index = 0; input_index < inputs; ++input_index)
            {
                Y[neuron_index][0] += X[input_index][0] * W[neuron_index][input_index];
            }
        }

        return Y;
    }

    /*
        Обратное распространение ошибки, подробности расчёта смотри в 
        самой функции
    */
    Matrix Backward(Matrix X, Matrix gradFromNextLayer)
    {
        if(gradFromNextLayer.getH() != outputs || gradFromNextLayer.getW() != 1)
        {
            std::cout << "Gradient from next layer is wrong dimension!" << std::endl;
            throw;
        }

        if(X.getH() != inputs || X.getW() != 1)
        {
            std::cout << "Input X is wrong dimension!" << std::endl;
            throw;
        }

        // Чтобы распространение ошибки продолжало работать, необходимо
        // передавать градиент дальше предыдущим слоям
        Matrix gradFromCurrLayer = Matrix(inputs, 1, 0);

        for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
        {
            for(unsigned int input_index = 0; input_index < inputs; ++input_index)
            {
                // Формула
                /*
                    Δw_i = δE/δS * δS/δw_i, где
                    E - ошибка слоя, то есть кто-то там впереди посчитал за нас, каким-то образом ошибку
                    S - фактический выход нейрона: S = b + w_1*x_1 + ... + w_i*x_i

                    Нам на слое предоставляется δE/δS - градиент со следующего слоя
                    Нам остается определить δS/δw, но это просто:

                    Если взять и продифференцировать S = b + w_1*x_1 + ... + w_i*x_i по какому-то w_j, 
                    то останется только x_j

                    Поэтому формула превращается в 
                    Δw_i = grad * x_i
                    Обрати внимание, что под grad понимается градиент, пришедший на нейрон,
                    Которому принадлежит w_i
                */

                /*
                    Считаем градиент для предыдущего слоя
                    Градиент для предыдущего слоя будет считаться так же просто
                    
                    Проговорим словами, что надо сделать
                    Для каждого входа слоя необходимо сосчитать сумму градиентов весов, 
                    Которые были применены к данному входу
                */
                gradFromCurrLayer[input_index][0] += gradFromNextLayer[neuron_index][0] * W[neuron_index][input_index];

                // Меняем веса на текущем слое
                W[neuron_index][input_index] -= gradFromNextLayer[neuron_index][0] * X[input_index][0] * learning_rate;
            }

            /*
                Как менять вес свободного члена?
                Продифференцируем S = b + w_1*x_1 + ... + w_i*x_i по b
                δS/δb = 1
                Поэтому Δb = grad
                Обрати внимание, что под grad понимается градиент, пришедший на нейрон,
                Которому принадлежит b
            */
            B[neuron_index][0] -= gradFromNextLayer[neuron_index][0] * learning_rate;
        }

        return gradFromCurrLayer;        
    }
    
};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif