#ifndef SOFTMAX_LAYER
#define SOFTMAX_LAYER

#include <cmath>
#include "Matrix.cpp"


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/*

    Этот слой был специально создан для использования совместно
    с функцией потерь Кросс-Энтропия

    Функция Softmax была выделена в отдельный класс, так как 
    рассчёт этой функции и особенно её градиента проходит не
    так, как у других функций активации.

    Вот тут хорошо рассказан смысл и рассчёт локального градиента 
    (хотя, в данном случае Якобиана):

    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    А здесь описан способ имплементации:

    https://habr.com/ru/companies/ods/articles/344116/

*/
class SoftmaxLayer
{

public:

    SoftmaxLayer(){}

    Matrix Forward(Matrix X)
    {
        if(X.getW() != 1)
        {
            std::cout << "Input X must be vertical vector!" << std::endl;
            throw;
        }

        Matrix Y = Matrix(X.getH(), 1);

        double S = 0;

        for(unsigned int i = 0; i < X.getH(); ++i)
            S += std::exp(X[i][0]);
        
        for(unsigned int i = 0; i < X.getH(); ++i)
            Y[i][0] = std::exp(X[i][0]) / S;
        
        return Y;
    }

    Matrix Backward(Matrix X, Matrix gradFromNextLayer)
    {
        Matrix Y = Matrix(X.getH(), 1);
        {
            double S = 0;

            for(unsigned int i = 0; i < X.getH(); ++i)
                S += std::exp(X[i][0]);
            
            for(unsigned int i = 0; i < X.getH(); ++i)
                Y[i][0] = std::exp(X[i][0]) / S;
        }        

        Matrix J = Matrix(X.getH(), X.getH());
        for(unsigned int i = 0; i < X.getH(); ++i)
        {
            for(unsigned int j = 0; j < X.getH(); ++j)
            {
                if (i == j)
                    J[i][j] = Y[i][0] * (1 - Y[i][0]);
                else
                    J[i][j] = -Y[i][0] * Y[j][0];
            }
        }

        Matrix gradFromCurrLayer = Matrix(X.getH(), 1);
        for(unsigned int i = 0; i < X.getH(); ++i)
        {
            for(unsigned int j = 0; j < X.getH(); ++j)
            {
                gradFromCurrLayer[i][0] += gradFromNextLayer[j][0] * J[j][i];
            }
        }

        return gradFromCurrLayer;
    }

};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#endif