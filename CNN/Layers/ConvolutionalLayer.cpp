#ifndef CONVOLUTIONAL_LAYER
#define CONVOLUTIONAL_LAYER

#include <vector>
#include <random>

#include "../Tensor.cpp"

using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class ConvolutionalLayer
{
    unsigned int input_depth;
    unsigned int input_height;
    unsigned int input_width;

    unsigned int filter_size;
    unsigned int filter_count;

    unsigned int padding;
    unsigned int stride;

    unsigned int output_height;
    unsigned int output_width;

public:

    double learning_rate;

    vector<Tensor> filters;
    vector<double> B;

    ConvolutionalLayer(){}

    void Initialize
    (
        unsigned int input_depth,
        unsigned int input_height,
        unsigned int input_width,
        unsigned int filter_size = 3,
        unsigned int filter_count = 1,
        unsigned int padding = 0,
        unsigned int stride = 1,
        double learning_rate = 1.0e-4,
        double mean = 0,
        double sigma = 0
    )
    {
        this->input_depth=input_depth;
        this->input_height=input_height;
        this->input_width=input_width;

        this->filter_size=filter_size;
        this->filter_count=filter_count;

        this->padding=padding;
        this->stride=stride;

        this->learning_rate=learning_rate;
        
        this->output_height = (input_height - filter_size + 2 * padding) / stride + 1;
        this->output_width = (input_width - filter_size + 2 * padding) / stride + 1;

        // Создание фильтров


        filters.reserve(filter_count);
        B.reserve(filter_count);
        for(unsigned int i = 0; i < filter_count; ++i)
        {      
            filters.push_back(Tensor(input_depth, filter_size, filter_size));
            B.push_back(0);
        }


        DefineWeights(sigma, mean);
    }


    void DefineWeights(double mean, double sigma = 0)
    {
        if (sigma == 0)
        {
            for(unsigned int f = 0; f < filter_count; ++f)
            {
                for(unsigned int d = 0; d < input_depth; ++d)
                {
                    for(unsigned int h = 0; h < filter_size; ++h)
                    {
                        for(unsigned int w = 0; w < filter_size; ++w)
                        {
                            filters[f][d][h][w] = mean;
                        }
                    }
                }
                B[f] = mean;
            }
        }
        else
        {
            // Создание генератора шума, распределенного нормально
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution d = std::normal_distribution(mean, sigma);
            auto random_double = [&d, &gen]{ return d(gen); };

            for(unsigned int f = 0; f < filter_count; ++f)
            {
                for(unsigned int d = 0; d < input_depth; ++d)
                {
                    for(unsigned int h = 0; h < filter_size; ++h)
                    {
                        for(unsigned int w = 0; w < filter_size; ++w)
                        {
                            filters[f][d][h][w] = random_double();
                        }
                    }
                }
                B[f] = random_double();
            }
        }

        

        
    }


    Tensor Forward(const Tensor& X) const
    {
        Tensor Y = Tensor(filter_count, output_height, output_width);

        for(unsigned int f = 0; f < filter_count; ++f)
        {

            for(unsigned int d = 0; d < input_depth; ++d)
            {
                
                for(unsigned int yh = 0; yh < output_height; ++yh)
                {
                    for(unsigned int yw = 0; yw < output_width; ++yw)
                    {
                        unsigned int i0 = yh * stride - padding;
                        unsigned int j0 = yw * stride - padding;

                        Y[f][yh][yw] = B[f];

                        for(unsigned int fh = 0; fh < filter_size; ++fh)
                        {
                            for(unsigned int fw = 0; fw < filter_size; ++fw)
                            {
                                
                                unsigned int i = fh + i0;
                                unsigned int j = fw + j0;
                            
                                if (i < 0 || i >= input_height || j < 0 || j >= input_width)
                                    continue;

                                Y[f][yh][yw] += filters[f](d, fh, fw) * X(d, i, j);
                            }
                        }
                    }
                }

            }

        }

        return Y;
    }


    Tensor Backward(const Tensor& X, const Tensor& GFNL)
    {
        
        // Обновление весов

        vector<Tensor> delta_w_f;
        delta_w_f.reserve(filter_count);

        for(unsigned int f = 0; f < filter_count; ++f)
        {
            delta_w_f.push_back(Tensor(input_depth, filter_size, filter_size, 0));
            
            for(unsigned int d = 0; d < input_depth; ++d)
            {
                for(unsigned int gh = 0; gh < output_height; ++gh)
                {
                    for(unsigned int gw = 0; gw < output_width; ++gw)
                    {
                        unsigned int h0 = gh * stride - padding;
                        unsigned int w0 = gw * stride - padding;

                        for(unsigned int fh = 0; fh < filter_size; ++fh)
                        {
                            for(unsigned int fw = 0; fw < filter_size; ++fw)
                            {
                                unsigned int h = h0 + fh;
                                unsigned int w = w0 + fw;

                                if (h < 0 || h >= input_height || w < 0 || w >= input_width)
                                    continue;
                                
                                delta_w_f[f][d][fh][fw] += GFNL(f, gh, gw) * X(d, h, w);
                            }
                        }
                    }
                }

            }
        }

        for(unsigned int f = 0; f < filter_count; ++f)
        {
            for(unsigned int d = 0; d < input_depth; ++d)
            {
                for(unsigned int h = 0; h < filter_size; ++h)
                {
                    for(unsigned int w = 0; w < filter_size; ++w)
                    {
                        filters[f][d][h][w] -= delta_w_f[f][d][h][w] * learning_rate; 
                    }
                }
            }

            double db = 0;

            for(unsigned int gh = 0; gh < output_height; ++gh)
            {
                for(unsigned int gw = 0; gw < output_width; ++gw)
                {
                    db += GFNL(f, gh, gw);
                }
            }

            B[f] -= db * learning_rate;
        }

        // Расчет возвращаемого градиента
        
        Tensor GFCL = Tensor(input_depth, input_height, input_width, 0);

        for(unsigned int f = 0; f < filter_count; ++f)
        {
            for(unsigned int d = 0; d < input_depth; ++d)
            {

                for(unsigned int gh = 0; gh < output_height; ++gh)
                {
                    for(unsigned int gw = 0; gw < output_width; ++gw)
                    {

                        unsigned int h0 = gh * stride - padding;
                        unsigned int w0 = gw * stride - padding;

                        for(unsigned int fh = 0; fh < filter_size; ++fh)
                        {
                            for(unsigned int fw = 0; fw < filter_size; ++fw)
                            {
                                unsigned int h = h0 + fh;
                                unsigned int w = w0 + fw;

                                if (h < 0 || h >= input_height || w < 0 || w >= input_width)
                                    continue;

                                GFCL[d][h][w] += GFNL(f, gh, gw) * filters[f](d, fh, fw);
                                
                            }
                        }

                    }   
                }

            }
        }

        return GFCL;
    }



};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#endif