#ifndef NET
#define NET

#include <string>
#include <random>
#include <vector>
#include <iostream>
#include <filesystem>

#include "Tensor.cpp"
#include "Layers/FullyConnectedLayer.cpp"
#include "Layers/ActivationLayer.cpp"
#include "Layers/SoftmaxLayer.cpp"

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
class Net
{

    Tensor FC1_X;
    Tensor AC1_X;

    Tensor FC2_X;
    Tensor AC2_X;

    Tensor FC3_X;
    Tensor AC3_X;

    FullyConnectedLayer fc1;
    ActivationLayer ac1;

    FullyConnectedLayer fc2;
    ActivationLayer ac2;

    FullyConnectedLayer fc3;
    SoftmaxLayer ac3;

    bool initialized = false;

public:

    Net(double learning_rate = 1e-4, double mean = 0, double sigma = 0.01)
    {
        fc1.Initialize(
            784,
            128,
            learning_rate,
            mean,
            sigma
        );
        ac1.SetActivationType(ActivationLayer::ActivationType::LeakyReLU);

        fc2.Initialize(
            128,
            64,
            learning_rate,
            mean,
            sigma
        );
        ac2.SetActivationType(ActivationLayer::ActivationType::Sigmoid);

        fc3.Initialize(
            64,
            10,
            learning_rate,
            mean,
            sigma
        );
        ac3 = SoftmaxLayer(); // Строка добавлена для общей красоты

        initialized = true;
    }
    
    Tensor Forward(Tensor& X)
    {
        FC1_X = X;
        
        AC1_X = fc1.Forward( FC1_X );
        FC2_X = ac1.Forward( AC1_X );

        AC2_X = fc2.Forward( FC2_X );
        FC3_X = ac2.Forward( AC2_X );

        AC3_X = fc3.Forward( FC3_X );
        Tensor Prediction = ac3.Forward( AC3_X );

        return Prediction;
    }


    unsigned int Predict(Tensor& X)
    {
        Tensor prediction = Forward(X);

        int predicted_number = 0;

        for(unsigned int j = 0; j < prediction.get_height(); ++j)
        {
            if (prediction[0][predicted_number][0] < prediction[0][j][0])
                predicted_number = j;
        }

        return predicted_number;
    }


    void Backward(Tensor& loss_gradient)
    {
        Tensor ac3_grad = ac3.Backward( AC3_X, loss_gradient);
        Tensor fc3_grad = fc3.Backward( FC3_X, ac3_grad);

        Tensor ac2_grad = ac2.Backward( AC2_X, fc3_grad);
        Tensor fc2_grad = fc2.Backward( FC2_X, ac2_grad);

        Tensor ac1_grad = ac1.Backward( AC1_X, fc2_grad);
        Tensor fc1_grad = fc1.Backward( FC1_X, ac1_grad);
    }

    /*
        Да, тут точность считается немного коряво, но это
        исключительно из-за формы возвращаемого предсказания

        Точность возвращается в процентах
    */
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


    void SetLearningRate(double lr)
    {
        fc1.learning_rate = lr;
        fc2.learning_rate = lr;
        fc3.learning_rate = lr;
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
        

        // Saving zero fully conn layer
        for(unsigned int d = 0; d < fc3.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc3.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc3.W.get_width(); ++w)
                    file << fc3.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fc3.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc3.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc3.B.get_width(); ++w)
                    file << fc3.B[d][h][w] << ' ';
        
        // Saving one fully conn layer
        for(unsigned int d = 0; d < fc2.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc2.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc2.W.get_width(); ++w)
                    file << fc2.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fc2.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc2.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc2.B.get_width(); ++w)
                    file << fc2.B[d][h][w] << ' ';
        
        // Saving two fully conn layer
        for(unsigned int d = 0; d < fc1.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc1.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc1.W.get_width(); ++w)
                    file << fc1.W[d][h][w] << ' ';
        
        for(unsigned int d = 0; d < fc1.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc1.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc1.B.get_width(); ++w)
                    file << fc1.B[d][h][w] << ' ';
        
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

        // Saving zero fully conn layer
        for(unsigned int d = 0; d < fc3.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc3.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc3.W.get_width(); ++w)
                    file >> fc3.W[d][h][w];
        
        for(unsigned int d = 0; d < fc3.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc3.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc3.B.get_width(); ++w)
                    file >> fc3.B[d][h][w];
        
        // Saving one fully conn layer
        for(unsigned int d = 0; d < fc2.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc2.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc2.W.get_width(); ++w)
                    file >> fc2.W[d][h][w];
        
        for(unsigned int d = 0; d < fc2.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc2.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc2.B.get_width(); ++w)
                    file >> fc2.B[d][h][w];
        
        // Saving two fully conn layer
        for(unsigned int d = 0; d < fc1.W.get_depth(); ++d)
            for(unsigned int h = 0; h < fc1.W.get_height(); ++h)
                for(unsigned int w = 0; w < fc1.W.get_width(); ++w)
                    file >> fc1.W[d][h][w];
        
        for(unsigned int d = 0; d < fc1.B.get_depth(); ++d)
            for(unsigned int h = 0; h < fc1.B.get_height(); ++h)
                for(unsigned int w = 0; w < fc1.B.get_width(); ++w)
                    file >> fc1.B[d][h][w];
        
        return true;
    }


};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif