#ifndef SIMPLE_NET_LAYER
#define SIMPLE_NET_LAYER

#include <string>
#include <random>
#include <vector>
#include "NetworkLayers/Matrix.cpp"
#include "NetworkLayers/FullyConnectedLayer.cpp"
#include "NetworkLayers/ActivationLayer.cpp"
#include "NetworkLayers/SoftmaxLayer.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


/*
    Простая сеть, архитектура которой выбрана почти случайным образом,
    но стоит пояснить одну вещей

    Если бы функция потерь Кросс-Энтропия хорошо работала с какой-либо
    функцией активации, кроме Softmax, то можно было бы и не писать
    слой активации SoftmaxLayer. Поэтому последним слоем обязательно 
    должен быть Softmax. 
    
    Однако, было бы естественно, если бы мы использовали функцию 
    активации Sigmoid, но почему-то градиент с этой функцией быстро 
    затухает, и сеть почти не обучается
*/
class SimpleNet
{

    Matrix FC1_X;
    Matrix AC1_X;

    Matrix FC2_X;
    Matrix AC2_X;

    Matrix FC3_X;
    Matrix AC3_X;

    FullyConnectedLayer fc1;
    ActivationLayer ac1;

    FullyConnectedLayer fc2;
    ActivationLayer ac2;

    FullyConnectedLayer fc3;
    SoftmaxLayer ac3;

public:

    SimpleNet(double learning_rate = 1e-4)
    {
        fc1 = FullyConnectedLayer(784, 128, learning_rate);
        ac1 = ActivationLayer(ActivationLayer::ActivationType::LeakyReLU);

        fc2 = FullyConnectedLayer(128, 64, learning_rate);
        ac2 = ActivationLayer(ActivationLayer::ActivationType::Sigmoid);

        fc3 = FullyConnectedLayer(64, 10, learning_rate);
        ac3 = SoftmaxLayer(); // Строка добавлена для общей красоты
    }

    
    Matrix Forward(Matrix X)
    {
        FC1_X = X;
        
        AC1_X = fc1.Forward(FC1_X);
        FC2_X = ac1.Forward(AC1_X);

        AC2_X = fc2.Forward(FC2_X);
        FC3_X = ac2.Forward(AC2_X);

        AC3_X = fc3.Forward(FC3_X);
        Matrix Prediction = ac3.Forward(AC3_X);

        return Prediction;
    }


    void Backward(Matrix loss_gradient)
    {
        Matrix ac3_grad = ac3.Backward(AC3_X, loss_gradient);
        Matrix fc3_grad = fc3.Backward(FC3_X, ac3_grad);

        Matrix ac2_grad = ac2.Backward(AC2_X, fc3_grad);
        Matrix fc2_grad = fc2.Backward(FC2_X, ac2_grad);

        Matrix ac1_grad = ac1.Backward(AC1_X, fc2_grad);
        Matrix fc1_grad = fc1.Backward(FC1_X, ac1_grad);
    }

    /*
        Да, тут точность считается немного коряво, но это
        исключительно из-за формы возвращаемого предсказания

        Точность возвращается в процентах
    */
    double Accuracy(vector<Matrix> X, vector<Matrix> Y, unsigned int check_sample_size = 0)
    {
        if (X.size() != Y.size())
        {
            cout << "Input data must have the same size!" << endl;
            throw;
        }

        unsigned int correct_count = 0;

        if (check_sample_size > 0)
        {
            for(unsigned int i = 0; i < check_sample_size; ++i)
            {
                unsigned int idx = rand() % X.size();

                Matrix prediction = Forward(X[idx]);

                int predicted_number = 0;

                for(unsigned int j = 0; j < prediction.getH(); ++j)
                {
                    if (prediction[predicted_number][0] < prediction[j][0])
                        predicted_number = j;
                }

                if (Y[idx][predicted_number][0] == 1)
                    ++correct_count;    
            }

            return ((double)correct_count) / ((double)check_sample_size) * 100;
        }
        else
        {
            for(unsigned int i = 0; i < X.size(); ++i)
            {
                Matrix prediction = Forward(X[i]);

                int predicted_number = 0;

                for(unsigned int j = 0; j < prediction.getH(); ++j)
                {
                    if (prediction[predicted_number][0] < prediction[j][0])
                        predicted_number = j;
                }

                if (Y[i][predicted_number][0] == 1)
                    ++correct_count;    
            }

            return ((double)correct_count) / ((double)X.size()) * 100;
        }
    }


    Matrix LossGradient(Matrix Y, Matrix Prediction)
    {
        if (Y.getW() != 1 || Prediction.getW() != 1)
        {
            cout << "Input data must be vertical vectors!"  << endl;
            throw;
        }

        if (Y.getH() != Prediction.getH())
        {
            cout << "Input data must have the same dimensions!"  << endl;
            throw;
        }

        Matrix grad = Matrix(Y.getH(), 1);

        for(unsigned int i = 0; i < Y.getH(); ++i)
        {
            grad[i][0] = Prediction[i][0] - Y[i][0];
        }

        return grad;
    }


    double Loss(vector<Matrix> X, vector<Matrix> Y, unsigned int check_sample_size = 0)
    {
        if (X.size() != Y.size())
        {
            cout << "Input data must have the same size!"  << endl;
            throw;
        }

        double L = 0;

        if (check_sample_size > 0)
        {
            for(unsigned int i = 0; i < X.size(); ++i)
            {
                unsigned int idx = rand() % X.size();
                L += Loss(Y[idx], Forward(X[idx]));
            }
            
            L /= check_sample_size;
        }
        else
        {
            for(unsigned int i = 0; i < X.size(); ++i)
            {
                L += Loss(Y[i], Forward(X[i]));
            }
                
            L /= X.size();
        }
        
        
        return L;
    }


    /*
        Бинарная кросс энтропия
    */
    double Loss(Matrix Y, Matrix Prediction)
    {
        if (Y.getW() != 1 || Prediction.getW() != 1)
        {
            cout << "Input data must be vertical vectors!"  << endl;
            throw;
        }

        if (Y.getH() != Prediction.getH())
        {
            cout << "Input data must have the same dimensions!"  << endl;
            throw;
        }

        double L = 0;

        for(unsigned int i = 0; i < Y.getH(); ++i)
            L -= Y[i][0] * log(Prediction[i][0]) + (1 - Y[i][0])*log(1 - Prediction[i][0]);
        
        return L;
    }

};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif