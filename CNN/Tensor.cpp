#ifndef TENSOR
#define TENSOR

#include <iostream>

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Tensor
{

private:

    double*** data = nullptr;
    unsigned int D = 0;
    unsigned int W = 0;
    unsigned int H = 0;

public:

// - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    Tensor()
    {
        data = nullptr;
        D = 0;
        W = 0;
        H = 0;
    }

    Tensor(const Tensor& other){

        if (D > 0 && H > 0 && W > 0)
        {
            for(unsigned int d = 0; d < D; ++d)
            {
                for(unsigned int h = 0; h < H; ++h)
                {
                    delete [] data[d][h];
                }
                delete [] data[d];
            }
            delete [] data;
        }
        
        D = other.D;
        H = other.H;
        W = other.W;
        
        data = new double**[D];
        for(unsigned int d = 0; d < D; ++d)
        {
            data[d] = new double*[H];
            for(unsigned int h = 0; h < H; ++h)
            {
                data[d][h] = new double[W];

                for(unsigned int w = 0; w < W; ++w)
                    data[d][h][w] = other.data[d][h][w];
            }
        }
    }

    Tensor(unsigned int D, unsigned int H, unsigned int W, double fill_val = 0)
    {
        if (D <= 0 || H <= 0 || W <= 0)
        {
            std::cout << "Height, width and depth of matrix must be greater then 0! " << D << 'x' << H << 'x' << W << std::endl;
            throw;
        }

        this->D = D;
        this->H = H;
        this->W = W;

        data = new double**[D];
        for(unsigned int d = 0; d < D; ++d)
        {
            data[d] = new double*[H];
            for(unsigned int h = 0; h < H; ++h)
            {
                data[d][h] = new double[W];

                for(unsigned int w = 0; w < W; ++w)
                    data[d][h][w] = fill_val;
            }
        }
    }

    ~Tensor()
    {
        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < H; ++h)
            {
                delete[] data[d][h];
            }
            delete[] data[d];
        }
        delete[] data;  

        D = 0;
        H = 0;
        W = 0; 
    }

// - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    const double operator()(unsigned int d, unsigned int h, unsigned int w) const
    {
        return data[d][h][w];
    }

    double** operator[](unsigned int d)
    {
        return data[d];
    }

    void operator/=(double val)
    {
        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < H; ++h)
            {
                for(unsigned int w = 0; w < W; ++w)
                {
                    data[d][h][w] /= val;
                }
            }
        }
    }

    void operator*=(double val)
    {
        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < H; ++h)
            {
                for(unsigned int w = 0; w < W; ++w)
                {
                    data[d][h][w] *= val;
                }
            }
        }
    }

    Tensor& operator=(const Tensor& other)
    {
        if (D > 0 && H > 0 && W > 0)
        {
            for(unsigned int d = 0; d < D; ++d)
            {
                for(unsigned int h = 0; h < H; ++h)
                {
                    delete [] data[d][h];
                }
                delete [] data[d];
            }
            delete [] data;
        }

        D = other.D;
        H = other.H;
        W = other.W;
        
        data = new double**[D];
        for(unsigned int d = 0; d < D; ++d)
        {
            data[d] = new double*[H];
            for(unsigned int h = 0; h < H; ++h)
            {
                data[d][h] = new double[W];

                for(unsigned int w = 0; w < W; ++w)
                    data[d][h][w] = other.data[d][h][w];
            }
        }

        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os,const Tensor& tensor) {

        os << tensor.D << 'x' << tensor.H << 'x' <<  tensor.W << std::endl;

        for (unsigned int d = 0; d < tensor.D; ++d) 
        {
            for (unsigned int h = 0; h < tensor.H; ++h) {
                
                for (unsigned int w = 0; w < tensor.W; ++w)
                    os << tensor.data[d][h][w] << " ";
                
                os << std::endl;
            }

            os << std::endl;
        }

        return os;
    }

// - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    void fill(double val)
    {
        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < H; ++h)
            {
                for(unsigned int w = 0; w < W; ++w)
                    data[d][h][w] = val;
            }
        }
    }

    unsigned int get_depth() const
    { 
        return D; 
    }

    unsigned int get_height() const
    {
        return H;
    }

    unsigned int get_width() const
    {
        return W;
    }

    void reshape(unsigned int newD, unsigned int newH, unsigned int newW)
    {
        if (newD * newH * newW != D * H * W)
        {
            std::cout << "Uncorrect shape for reshaping!" << std::endl;
            throw;
        }

        double* arr = new double[D * H * W];

        for(unsigned int d = 0; d < D; ++d)
            for(unsigned int h = 0; h < H; ++h)
                for(unsigned int w = 0; w < W; ++w)
                    arr[d * H * W + h * W + w] = data[d][h][w];
        
        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < H; ++h)
            {
                delete [] data[d][h];
            }
            delete [] data[d];
        }

        delete [] data;

        D = newD;
        H = newH;
        W = newW;

        data = new double**[D];
        for(unsigned int d = 0; d < D; ++d)
        {
            data[d] = new double*[H];
            for(unsigned int h = 0; h < H; ++h)
            {
                data[d][h] = new double[W];
            }
        }
        
        for(unsigned int d = 0; d < D; ++d)
            for(unsigned int h = 0; h < H; ++h)
                for(unsigned int w = 0; w < W; ++w)
                    data[d][h][w] = arr[d * H * W + h * W + w];
    }
};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif