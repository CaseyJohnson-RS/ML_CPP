#ifndef MAX_POOL_LAYER
#define MAX_POOL_LAYER

#include <iostream>
#include "../Tensor.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class MaxPoolLayer
{
    unsigned int D;

    unsigned int XH;
    unsigned int XW;

    unsigned int YH;
    unsigned int YW;

    unsigned int kernel_size = 0;
    unsigned int kernel_stride = 0;

    unsigned int*** GHIndex;
    unsigned int*** GWIndex;

public:

    MaxPoolLayer(){}


    void Initialize(unsigned int depth, unsigned int height, unsigned int width, unsigned int pool_size = 3, unsigned int stride = 1)
    {
        kernel_size = pool_size;
        kernel_stride = stride;

        D = depth;

        XH = height;
        XW = width;

        YH = (XH - kernel_size) / kernel_stride + 1;
        YW = (XW - kernel_size) / kernel_stride + 1;

        GHIndex = new unsigned int**[D];
        GWIndex = new unsigned int**[D];
        for(unsigned int d = 0; d < D; ++d)
        {
            GHIndex[d] = new unsigned int*[YH];
            GWIndex[d] = new unsigned int*[YH];
            for(unsigned int h = 0; h < YH; ++h)
            {
                GHIndex[d][h] = new unsigned int[YW];
                GWIndex[d][h] = new unsigned int[YW];
                for(unsigned int w = 0; w < YW; ++w)
                {
                    GHIndex[d][h][w] = 0;
                    GWIndex[d][h][w] = 0;
                }
            }
        }
    }


    Tensor Forward(const Tensor& X) 
    {
        if (X.get_height() != XH || X.get_width() != XW || X.get_depth() != D)
        {
            std::cout << "Input tensor is wrong size (Max pool layer)!" << std::endl;
            throw;
        }

        Tensor Y = Tensor(D, YH, YW);

        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < YH; ++h)
            {
                for(unsigned int w = 0; w < YW; ++w)
                {
                    unsigned int window_top = kernel_stride * h;
                    unsigned int window_bottom = window_top + kernel_size;
                    unsigned int window_left = kernel_stride * w;
                    unsigned int window_right = window_left + kernel_size;

                    Y[d][h][w] = X(d, window_top, window_left);
                    GHIndex[d][h][w] = window_top;
                    GWIndex[d][h][w] = window_left;

                    for(unsigned int kh = window_top; kh < window_bottom; ++kh)
                    {
                        for(unsigned int kw = window_left; kw < window_right; ++kw)
                        {
                            if (Y(d, h, w) < X(d, kh, kw))
                            {
                                Y[d][h][w] = X(d, kh, kw);
                                GHIndex[d][h][w] = kh;
                                GWIndex[d][h][w] = kw;
                            }
                        }
                    }
                }
            }
        }

        return Y;
    }

    Tensor Backward(Tensor& GFNL) // GNFL - gradient from next layer
    {
        Tensor GFCL = Tensor(D, XH, XW);

        for(unsigned int d = 0; d < D; ++d)
        {
            for(unsigned int h = 0; h < YH; ++h)
            {
                for(unsigned int w = 0; w < YW; ++w)
                {
                    GFCL[d][GHIndex[d][h][w]][GWIndex[d][h][w]] = GFNL(d, h, w);
                }
            }
        }

        return GFCL;
    }

};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#endif