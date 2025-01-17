#ifndef SOFTMAX_LAYER
#define SOFTMAX_LAYER

#include <cmath>
#include "../Tensor.cpp"


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

    Tensor Forward(const Tensor& X)
    {
        if(X.get_depth() != 1 || X.get_width() != 1)
        {
            std::cout << "Input X must be vertical vector!" << std::endl;
            throw;
        }

        Tensor Y = Tensor(1, X.get_height(), 1);

        double S = 0;

        for(unsigned int i = 0; i < X.get_height(); ++i)
            S += std::exp(X(0, i, 0));
        
        for(unsigned int i = 0; i < X.get_height(); ++i)
            Y[0][i][0] = std::exp(X(0, i, 0)) / S;
        
        return Y;
    }


    Tensor Backward(const Tensor& X, const Tensor& GFNL)
    {
        Tensor Y = Tensor(1, X.get_height(), 1);
        {
            double S = 0;

            for(unsigned int i = 0; i < X.get_height(); ++i)
                S += std::exp(X(0, i, 0));
            
            for(unsigned int i = 0; i < X.get_height(); ++i)
                Y[0][i][0] = std::exp(X(0, i, 0)) / S;
        }        

        Tensor J = Tensor(1, X.get_height(), X.get_height());
        for(unsigned int i = 0; i < X.get_height(); ++i)
        {
            for(unsigned int j = 0; j < X.get_height(); ++j)
            {
                if (i == j)
                    J[0][i][j] = Y(0, i, 0) * (1 - Y(0, i, 0));
                else
                    J[0][i][j] = -Y(0, i, 0) * Y(0, i, 0);
            }
        }

        Tensor GFCL = Tensor(1, X.get_height(), 1);
        for(unsigned int i = 0; i < X.get_height(); ++i)
        {
            for(unsigned int j = 0; j < X.get_height(); ++j)
            {
                GFCL[0][i][0] += GFNL(0, j, 0) * J(0, j, i);
            }
        }

        return GFCL;
    }

};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#endif