
#ifndef AVG_POOL_LAYER
#define AVG_POOL_LAYER

#include <iostream>
#include "../Tensor.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class AvgPoolLayer
{
    unsigned int XD;
    unsigned int XH;
    unsigned int XW;

    unsigned int YD;
    unsigned int YH;
    unsigned int YW;

    unsigned int kernel_size = 0;
    unsigned int kernel_stride = 0;
    unsigned int kernel_size_squared = 0;

public:

    AvgPoolLayer(){}

    void Initialize(unsigned int input_tensor_depth, unsigned int input_tensor_height, unsigned int input_tensor_width, unsigned int size = 3, unsigned int stride = 1)
    {
        kernel_size = size;
        kernel_stride = stride;

        XD = input_tensor_depth;
        XH = input_tensor_height;
        XW = input_tensor_width;

        YD = input_tensor_depth;
        YH = (XH - kernel_size) / kernel_stride + 1;
        YW = (XW - kernel_size) / kernel_stride + 1;

        kernel_size_squared = kernel_size * kernel_size;
    }

    Tensor Forward(const Tensor& X) 
    {
        if (X.get_height() != XH || X.get_width() != XW || X.get_depth() != XD)
        {
            std::cout << "Input tensor is wrong size (Avg pool layer)!" << std::endl;
            throw;
        }

        Tensor Y = Tensor(YD, YH, YW);
        
        for(unsigned int d = 0; d < YD; ++d)
        {
            for(unsigned int h = 0; h < YH; ++h)
            {
                for(unsigned int w = 0; w < YW; ++w)
                {
                    unsigned int window_top = kernel_stride * h;
                    unsigned int window_bottom = window_top + kernel_size;
                    unsigned int window_left = kernel_stride * w;
                    unsigned int window_right = window_left + kernel_size;

                    for(unsigned int kh = window_top; kh < window_bottom; ++kh)
                    {
                        for(unsigned int kw = window_left; kw < window_right; ++kw)
                        {
                            Y[d][h][w] += X(d, kh, kw);
                        }
                    }

                    Y[d][h][w] /= kernel_size_squared;
                }
            }
        }

        return Y;
    }

    Tensor Backward(const Tensor& GFNL) // GNFL - gradient from next layer
    {
        if (GFNL.get_height() != YH || GFNL.get_width() != YW || GFNL.get_depth() != YD)
        {
            std::cout << "Input tensor is wrong size (Avg pool layer)!" << std::endl;
            throw;
        }

        Tensor GFCL = Tensor(XD, XH, XW);

        for(unsigned int d = 0; d < YD; ++d)
        {
            for(unsigned int h = 0; h < YH; ++h)
            {
                for(unsigned int w = 0; w < YW; ++w)
                {
                    unsigned int window_top = kernel_stride * h;
                    unsigned int window_bottom = window_top + kernel_size;
                    unsigned int window_left = kernel_stride * w;
                    unsigned int window_right = window_left + kernel_size;

                    for(unsigned int kh = window_top; kh < window_bottom; ++kh)
                    {
                        for(unsigned int kw = window_left; kw < window_right; ++kw)
                        {
                            GFCL[d][kh][kw] += GFNL(d, h, w);
                        }
                    }

                }
            }
        }

        GFCL /= kernel_size_squared;

        return GFCL;
    }

};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#endif