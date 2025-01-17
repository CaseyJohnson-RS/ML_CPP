#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include <iostream>
#include <random>

#include "../Tensor.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/*
    Полносвязный слой, разобраться в котором не так сложно:

    Самая сложная часть:
    
    https://education.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki
*/

class FullyConnectedLayer
{

private:

    unsigned int inputs;
    unsigned int outputs;

public:

    double learning_rate;

    Tensor W;
    Tensor B;

    FullyConnectedLayer(){}

    void Initialize(
        unsigned int inputs, 
        unsigned int outputs, 
        double learning_rate = 1e-4,
        double mean = 0,
        double sigma = 0
    )
    {
        this -> inputs = inputs;
        this -> outputs = outputs;
        this -> learning_rate = learning_rate;

        W = Tensor(1, outputs, inputs);
        B = Tensor(1, outputs, 1);

        DefineWeights(mean, sigma);
    }

    /*
        Инициализация весов с помощью нормального распределения
    */
    void DefineWeights(double mean, double sigma = 0)
    {

        if (sigma == 0)
        {
            for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
            {
                B[0][neuron_index][0] = mean;
                for(unsigned int input_index = 0; input_index < inputs; ++input_index)
                {
                    W[0][neuron_index][input_index] = mean;
                }
            }
        }
        else 
        {
            // Создание генератора шума, распределенного нормально
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution d = std::normal_distribution(mean, sigma);
            auto random_double = [&d, &gen]{ return d(gen); };

            for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
            {
                B[0][neuron_index][0] = random_double();
                for(unsigned int input_index = 0; input_index < inputs; ++input_index)
                {
                    W[0][neuron_index][input_index] = random_double();
                }
            }
        }
    }


    /*
        Банальное перемножение весов каждого нейрона со входным сигналом
        и суммирование
    */
    Tensor Forward(const Tensor& X)
    {
        if (X.get_height() != inputs || X.get_width() != 1)
        {
            std::cout << "Input X must be vertical vector with depth = 1!" << std::endl;
            throw;
        }
        
        Tensor Y = Tensor(1, outputs, 1);

        for(unsigned int neuron_index = 0; neuron_index < outputs; ++neuron_index)
        {
            Y[0][neuron_index][0] = B[0][neuron_index][0];

            for(unsigned int input_index = 0; input_index < inputs; ++input_index)
            {
                Y[0][neuron_index][0] += X(0, input_index, 0) * W(0, neuron_index, input_index);
            }
        }

        return Y;
    }

    /*
        Обратное распространение ошибки, подробности расчёта смотри в 
        самой функции
    */
    Tensor Backward(const Tensor& X, const Tensor& GFNL)
    {
        if(GFNL.get_height() != outputs || GFNL.get_width() != 1 || GFNL.get_depth() != 1)
        {
            std::cout << "Gradient from next layer is wrong dimension!" << std::endl;
            throw;
        }

        if(X.get_height() != inputs || X.get_width() != 1 || X.get_depth() != 1)
        {
            std::cout << "Input X is wrong dimension!" << std::endl;
            throw;
        }

        // Чтобы распространение ошибки продолжало работать, необходимо
        // передавать градиент дальше предыдущим слоям
        Tensor GFCL = Tensor(1, inputs, 1);

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
                GFCL[0][input_index][0] += GFNL(0, neuron_index, 0) * W(0, neuron_index, input_index);

                // Меняем веса на текущем слое
                W[0][neuron_index][input_index] -= GFNL(0, neuron_index, 0) * X(0, input_index, 0) * learning_rate;
            }

            /*
                Как менять вес свободного члена?
                Продифференцируем S = b + w_1*x_1 + ... + w_i*x_i по b
                δS/δb = 1
                Поэтому Δb = grad
                Обрати внимание, что под grad понимается градиент, пришедший на нейрон,
                Которому принадлежит b
            */
            B[0][neuron_index][0] -= GFNL(0, neuron_index, 0) * learning_rate;
        }

        return GFCL;        
    }
    
};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif