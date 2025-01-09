#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <string>

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/*
    Это очень глупенький класс матрицы

    Матричное перемножение (не поэлементное!)

    C = A * B;

    Вывод в поток

    cout << A;
*/
class Matrix
{

private:

    double** data;
    unsigned int W;
    unsigned int H;

public:

    Matrix()
    {
        this->data = nullptr;
        this->W = 0;
        this->H = 0;
    }

    Matrix(unsigned int h, unsigned int w, double fill = 0)
    {
        if (h <= 0 || w <= 0)
        {
            std::cout << "Height and width of matrix must be greater then 0!" << std::endl;
            throw;
        }

        H = h;
        W = w;

        data = new double*[H];
        for(unsigned int i = 0; i < H; ++i)
        {
            data[i] = new double[W];

            for(unsigned int j = 0; j < W; ++j)
                data[i][j] = fill;
        }
    }

    unsigned int getH(){ return H; }
    unsigned int getW(){ return W; }

    double* operator[] (int index) { return data[index]; }

    Matrix operator* (Matrix b)
    {
        if (W != b.getH())
        {
            std::cout << "Wrong matrix dimensions!" << std::endl;
            throw;
        }
            
        Matrix c = Matrix(H, b.getW());
        
        for(unsigned int ha = 0; ha < H; ++ha)
            for(unsigned int wb = 0; wb < b.getW(); ++wb)
                for(unsigned int k = 0; k < W; ++k)
                    c[ha][wb] += data[ha][k] * b[k][wb];

        return c;
    }

    friend std::ostream& operator<<(std::ostream& os, Matrix obj) 
    {

        for(unsigned int i = 0; i < obj.getH(); ++i)
        {
            for(unsigned int j = 0; j < obj.getW(); ++j)
            {
                os << obj[i][j] << "\t";
            }
            os << std::endl;
        }

        return os;
    }

};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif