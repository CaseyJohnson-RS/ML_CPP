#ifndef LENET_MNIST
#define LENET_MNIST

#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include "Tensor.cpp"
#include "Layers/MaxPoolLayer.cpp"
#include "Layers/AvgPoolLayer.cpp"
#include "Layers/ConvolutionalLayer.cpp"
#include "Layers/ActivationLayer.cpp"
#include "Layers/FullyConnectedLayer.cpp"
#include "Layers/SoftmaxLayer.cpp"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class LeNetMnist
{
    ConvolutionalLayer convl_0;
    MaxPoolLayer mpl_0;
    ActivationLayer convactl_0;

    ConvolutionalLayer convl_1;
    MaxPoolLayer mpl_1;
    ActivationLayer convactl_1;

    // -- 

    FullyConnectedLayer fcl_0;
    ActivationLayer actl_0;

    FullyConnectedLayer fcl_1;
    ActivationLayer actl_1;

    FullyConnectedLayer fcl_2;
    SoftmaxLayer sml;

    // - - - -

    Tensor CL_0_X;
    Tensor MPL_0_X;
    Tensor CONVACTL_0_X;
    
    Tensor CL_1_X;
    Tensor MPL_1_X;
    Tensor CONVACTL_1_X;

    Tensor FCL_0_X;
    Tensor ACTL_0_X;

    Tensor FCL_1_X;
    Tensor ACTL_1_X;

    Tensor FCL_2_X;
    Tensor SML_X;

    bool initialized = false;

public:

    LeNetMnist(double learning_rate = 1.0E-4, double mean = 0, double sigma = 0)
    {
        convl_0.Initialize(
            1,                  // input channels
            28,                 // image height
            28,                 // image width
            5,                  // filter kernel size
            6,                  // amount of filters
            0,                  // padding
            1,                  // stride
            learning_rate,      // learning rate
            mean,               // mean
            sigma               // sigma
        );
        
        mpl_0.Initialize(
            6,                  // input channels
            24,                 // image height
            24,                 // image width
            2,                  // pool size
            2                   // stride
        );

        convactl_0.SetActivationType(ActivationLayer::ActivationType::Tanh);


        convl_1.Initialize(
            6,                  // input channels
            12,                 // image height
            12,                 // image width
            5,                  // filter kernel size
            16,                 // amount of filters
            0,                  // padding
            1,                  // stride
            learning_rate,      // learning rate
            mean,               // mean
            sigma               // sigma
        );
        
        mpl_1.Initialize(
            16,                 // input channels
            8,                 // image height
            8,                 // image width
            2,                  // pool size
            2                   // stride
        );

        convactl_1.SetActivationType(ActivationLayer::ActivationType::Tanh);

        // -- 

        fcl_0.Initialize(
            256,
            120,
            learning_rate,
            mean,
            sigma
        );
        
        actl_0.SetActivationType(ActivationLayer::ActivationType::Tanh);


        fcl_1.Initialize(
            120,
            84,
            learning_rate,
            mean,
            sigma
        );
        actl_1.SetActivationType(ActivationLayer::ActivationType::Tanh);

        
        fcl_2.Initialize(
            84,
            10,
            learning_rate,
            mean,
            sigma
        );

        // And Softmaxlayer (sml)

        initialized = true;
    }


    unsigned int get_height_index_of_maximum_in_tensor(const Tensor& X)
    {
        unsigned int index = 0;

        for(unsigned int j = 0; j < X.get_height(); ++j)
        {
            if (X(0, index, 0) < X(0, j, 0))
                index = j;
        }

        return index;
    }


    unsigned int Predict(Tensor& X)
    {
        return get_height_index_of_maximum_in_tensor(Forward(X));
    }


    Tensor Forward(Tensor& X)
    {
        CL_0_X = X;
        
        MPL_0_X = convl_0.Forward(CL_0_X);
        CONVACTL_0_X = mpl_0.Forward(MPL_0_X);
        CL_1_X = convactl_0.Forward(CONVACTL_0_X);

        MPL_1_X = convl_1.Forward(CL_1_X);
        CONVACTL_1_X = mpl_1.Forward(MPL_1_X);
        FCL_0_X = convactl_1.Forward(CONVACTL_1_X);

        FCL_0_X.reshape(1, 256, 1);

        ACTL_0_X = fcl_0.Forward(FCL_0_X);
        FCL_1_X = actl_0.Forward(ACTL_0_X);

        ACTL_1_X = fcl_1.Forward(FCL_1_X);
        FCL_2_X = actl_1.Forward(ACTL_1_X);

        SML_X = fcl_2.Forward(FCL_2_X);
        Tensor Y = sml.Forward(SML_X);

        return Y;
    }


    void Backward(const Tensor& dL)
    {
        Tensor sml_grad = sml.Backward(SML_X, dL);
        Tensor fcl_2_grad = fcl_2.Backward(FCL_2_X, sml_grad);
        
        Tensor actl_1_grad = actl_1.Backward(ACTL_1_X, fcl_2_grad);
        Tensor fcl_1_grad = fcl_1.Backward(FCL_1_X, actl_1_grad);

        Tensor actl_0_grad = actl_0.Backward(ACTL_0_X, fcl_1_grad);
        Tensor fcl_0_grad = fcl_0.Backward(FCL_0_X, actl_0_grad);

        fcl_0_grad.reshape(16, 4, 4);

        Tensor convactl_1_grad = convactl_1.Backward(CONVACTL_1_X, fcl_0_grad);
        Tensor mpl_1_grad = mpl_1.Backward(convactl_1_grad);
        Tensor convl_1_grad = convl_1.Backward(CL_1_X, mpl_1_grad);

        Tensor convactl_0_grad = convactl_0.Backward(CONVACTL_0_X, convl_1_grad);
        Tensor mpl_0_grad = mpl_0.Backward(convactl_0_grad);
        Tensor convl_0_grad = convl_0.Backward(CL_0_X, mpl_0_grad);

// #include <iostream>
// std::cout << std::endl << "mpl_0_grad" << std::endl << mpl_0_grad << std::endl;
    }


    double Accuracy(vector<Tensor> X, vector<unsigned int> Y, unsigned int check_sample_size = 0)
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

                if (Y[idx] == Predict(X[idx]))
                    ++correct_count;
            }

            return ((double)correct_count) / ((double)check_sample_size) * 100;
        }
        else
        {
            for(unsigned int i = 0; i < X.size(); ++i)
            {
                if (Y[i] == Predict(X[i]))
                    ++correct_count;    
            }

            return ((double)correct_count) / ((double)X.size()) * 100;
        }
    }


    Tensor LossGradient(unsigned int Y, Tensor prediction)
    {
        if (prediction.get_width() != 1 || prediction.get_depth() != 1)
        {
            cout << "Input data must be vertical vectors!"  << endl;
            throw;
        }

        Tensor grad = Tensor(1, prediction.get_height(), 1);

        for(unsigned int i = 0; i < prediction.get_height(); ++i)
        {
            if (i == Y)
            {
                grad[0][i][0] = prediction(0, i, 0) - 1;
            }
            else
            {
                grad[0][i][0] = prediction(0, i, 0);
            }
        }

        return grad;
    }


    double Loss(vector<Tensor> X, vector<unsigned int> Y, unsigned int check_sample_size = 0)
    {
        if (X.size() != Y.size())
        {
            cout << "Input data must have the same size!"  << endl;
            throw;
        }

        double L = 0;

        if (check_sample_size > 0)
        {
            for(unsigned int i = 0; i < check_sample_size; ++i)
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
    double Loss(unsigned int Y, Tensor prediction)
    {
        if (prediction.get_width() != 1 || prediction.get_depth() != 1)
        {
            cout << "Input data must be vertical vectors!"  << endl;
            throw;
        }

        double L = -log(prediction(0, Y, 0));

        for(unsigned int i = 0; i < prediction.get_height(); ++i)
        {
            if (i != Y)
                L -= log(1 - prediction(0, i, 0));
        }
        
        return L;
    }


    bool SaveModel(std::string name)
    {
        if (!initialized)
            return false;

        std::string dir = "Models";

        if (!std::filesystem::exists(dir)) {
            if (!std::filesystem::create_directories(dir))
                return false;
        }

        std::string path = dir + "/" + name + ".mdl";

        std::ofstream file;
        file.open (path);

        // Saving zero conv layer
        for(unsigned int f = 0; f < convl_0.filters.size(); ++f)
            for(unsigned int d = 0; d < convl_0.filters[0].get_depth(); ++d)
                for(unsigned int h = 0; h < convl_0.filters[0].get_height(); ++h)
                    for(unsigned int w = 0; w < convl_0.filters[0].get_width(); ++w)
                        file << convl_0.filters[f][d][h][w] << ' ';
        
        for(unsigned int f = 0; f < convl_0.B.size(); ++f)
            file << convl_0.B[f] << ' ';

        // Saving one conv layer
        for(unsigned int f = 0; f < convl_1.filters.size(); ++f)
            for(unsigned int d = 0; d < convl_1.filters[0].get_depth(); ++d)
                for(unsigned int h = 0; h < convl_1.filters[0].get_height(); ++h)
                    for(unsigned int w = 0; w < convl_1.filters[0].get_width(); ++w)
                        file << convl_1.filters[f][d][h][w] << ' ';
        
        for(unsigned int f = 0; f < convl_1.B.size(); ++f)
            file << convl_1.B[f] << ' ';
        

        // Saving zero fully conn layer
        for(unsigned int d = 0; d < fcl_0.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_0.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_0.W.get_width(); ++w)
                    file << fcl_0.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fcl_0.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_0.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_0.B.get_width(); ++w)
                    file << fcl_0.B[d][h][w] << ' ';
        
        // Saving one fully conn layer
        for(unsigned int d = 0; d < fcl_1.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_1.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_1.W.get_width(); ++w)
                    file << fcl_1.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fcl_1.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_1.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_1.B.get_width(); ++w)
                    file << fcl_1.B[d][h][w] << ' ';
        
        // Saving two fully conn layer
        for(unsigned int d = 0; d < fcl_2.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_2.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_2.W.get_width(); ++w)
                    file << fcl_2.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fcl_2.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_2.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_2.B.get_width(); ++w)
                    file << fcl_2.B[d][h][w] << ' ';
        
        return true;
    }

    bool ReadModel(std::string name)
    {
        if (!initialized)
            return false;

        std::string dir = "Models";

        if (!std::filesystem::exists(dir)) 
            return false;

        std::string path = dir + "/" + name + ".mdl";

        std::ifstream file;
        file.open(path);

        if (!file.is_open())
            return false;

        // Saving zero conv layer
        for(unsigned int f = 0; f < convl_0.filters.size(); ++f)
            for(unsigned int d = 0; d < convl_0.filters[0].get_depth(); ++d)
                for(unsigned int h = 0; h < convl_0.filters[0].get_height(); ++h)
                    for(unsigned int w = 0; w < convl_0.filters[0].get_width(); ++w)
                        file >> convl_0.filters[f][d][h][w];
        
        for(unsigned int f = 0; f < convl_0.B.size(); ++f)
            file >> convl_0.B[f];

        // Saving one conv layer
        for(unsigned int f = 0; f < convl_1.filters.size(); ++f)
            for(unsigned int d = 0; d < convl_1.filters[0].get_depth(); ++d)
                for(unsigned int h = 0; h < convl_1.filters[0].get_height(); ++h)
                    for(unsigned int w = 0; w < convl_1.filters[0].get_width(); ++w)
                        file >> convl_1.filters[f][d][h][w];
        
        for(unsigned int f = 0; f < convl_1.B.size(); ++f)
            file >> convl_1.B[f];
        

        // Saving zero fully conn layer
        for(unsigned int d = 0; d < fcl_0.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_0.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_0.W.get_width(); ++w)
                    file >> fcl_0.W[d][h][w];
        
        for(unsigned int d = 0; d < fcl_0.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_0.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_0.B.get_width(); ++w)
                    file >> fcl_0.B[d][h][w];
        
        // Saving one fully conn layer
        for(unsigned int d = 0; d < fcl_1.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_1.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_1.W.get_width(); ++w)
                    file >> fcl_1.W[d][h][w];
        
        for(unsigned int d = 0; d < fcl_1.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_1.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_1.B.get_width(); ++w)
                    file >> fcl_1.B[d][h][w];
        
        // Saving two fully conn layer
        for(unsigned int d = 0; d < fcl_2.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_2.W.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_2.W.get_width(); ++w)
                    file >> fcl_2.W[d][h][w];
        
        for(unsigned int d = 0; d < fcl_2.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fcl_2.B.get_height(); ++h)
                for(unsigned int w = 0; w < fcl_2.B.get_width(); ++w)
                    file >> fcl_2.B[d][h][w];
        
        return true;
    }
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif