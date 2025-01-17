#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <random>
#include <algorithm>

#include "Tensor.cpp"
#include "LeNet.cpp"

using namespace std;


void TrainCNN_CIFAR10()
{
    cout << endl << endl <<  "- - - - - - - - - - - - - - - - - - - - - " << endl << endl;
    cout << "\tTrain Cifar 10" << endl << endl;
    cout << "- - - - - - - - - - - - - - - - - - - - - " << endl << endl;

    srand(time(0));

    double learning_rate = 0.001;
    double mean = 0;
    double sigma = 0.01;
    bool mini_batch_mode = true;
    bool show_start_accuracy = false;
    string load_model = "";

    cout << endl << "Learning rate: "; 
    cin >> learning_rate;
    cout << "Weights mean: "; 
    cin >> mean;
    cout << "Weights sigma: "; 
    cin >> sigma;
    cout << "Learning mode (minibatch - 1, entire sample - 0): "; 
    cin >> mini_batch_mode;
    cout << "Load model (n - no model): "; 
    cin >> load_model;
    cout << "Show start accuracy (1 - yes, 0 - no): "; 
    cin >> show_start_accuracy;
    cout << endl;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    string path = "data/";

    unsigned int batch_size = 10000;

    cout << "> Reading data... ";  

    // 50000
    vector<Tensor> train_images;
    vector<unsigned int> train_labels;

    // 10000
    vector<Tensor> test_images;
    vector<unsigned int> test_labels;

    // Чтение тренировочных картинок

    train_images.reserve(batch_size * 5);
    train_labels.reserve(batch_size * 5);
    for(unsigned int i = 1; i <= 5; ++i)
    {
        ifstream inputFileStream(path + "data_batch_" + to_string(i) + ".bin", ios::in | std::ios::binary);    
        if (inputFileStream.fail()) 
            throw std::runtime_error("Failed to open " + path + "data_batch_" + to_string(i) + ".bin");
        
        byte cursor;
        
        for(int i = 0; i < batch_size; ++i)
        {
            inputFileStream.read((char*)&cursor, sizeof(cursor) );

            train_labels.push_back((int)cursor);

            Tensor X = Tensor(3, 32, 32);

            for(unsigned int d = 0; d < 3; ++d)
            {
                for(unsigned int h = 0; h < 32; ++h)
                {
                    for(unsigned int w = 0; w < 32; ++w)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor) );
                        
                        X[d][h][w] = (double)cursor / 255 - 0.5;
                    }
                }
            }

            train_images.push_back(X);
        }

        inputFileStream.close();
    }

    // Чтение тестировочных картинок

    test_images.reserve(batch_size);
    test_labels.reserve(batch_size);
    {
        ifstream inputFileStream(path + "test_batch.bin", ios::in | std::ios::binary);    
        if (inputFileStream.fail()) 
            throw std::runtime_error("Failed to open " + path + "test_batch.bin");
        
        byte cursor;
        
        for(int i = 0; i < batch_size; ++i)
        {
            inputFileStream.read((char*)&cursor, sizeof(cursor) );

            test_labels.push_back((int)cursor);

            Tensor X = Tensor(3, 32, 32);

            for(unsigned int d = 0; d < 3; ++d)
            {
                for(unsigned int h = 0; h < 32; ++h)
                {
                    for(unsigned int w = 0; w < 32; ++w)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor) );
                        
                        X[d][h][w] = (double)cursor / 255 - 0.5;
                    }
                }
            }

            test_images.push_back(X);
        }

        inputFileStream.close();
    }

    cout << "done!" << endl;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    LeNet net = LeNet(learning_rate, mean, sigma);

    // Загрузка уже предобученной модели
    if (load_model != "n")
    {
        cout << "> Try load model '" << load_model << "'... ";
        cout << (net.ReadModel(load_model) ? "model is loaded" : "failed to load model...") << endl;
    }

    if (show_start_accuracy)
        cout << "> Start accuracy: " << net.Accuracy(test_images, test_labels, 1000) << endl;
    
    // Обучение одном батче в эпоху
    if (mini_batch_mode)
    {
        unsigned int eras = 50;

        cout << endl << "> Mini batch learning mode" << endl;
        cout << endl << "Epoch\tProgress\tAccuracy\tLoss(Cross Entropy)" << endl;

        for(unsigned int epoch = 0; epoch < eras; ++epoch)
        {
            cout << epoch + 1 << "/" << eras << "\t";

            unsigned int batch_idx = rand() % 5;

            for(unsigned int j = 0; j < batch_size; ++j)
            {
                Tensor prediction = net.Forward(train_images[j + batch_size * batch_idx]);
                Tensor gradient = net.LossGradient(train_labels[j + batch_size * batch_idx], prediction);
                net.Backward(gradient);

                if (j % (batch_size / 10) == 0)
                    cout << "-";
            }

            double loss = net.Loss(test_images, test_labels, 1000);
            double accur = net.Accuracy(test_images, test_labels, 1000);

            cout << "\t" << accur << '%' << "\t\t" << loss;
            cout << "\t\t" << (net.SaveModel("avg_" + to_string((int)accur) + '.' + to_string((int)(100 * accur) % 100)) ? "\tModel saved" : "\tFailed to save model...") << endl;
        }
    }

    // Обучение на всей выборке
    if (!mini_batch_mode)
    {
        unsigned int eras = 50;

        cout << endl << "> All batch learning mode" << endl;
        cout << endl << "Epoch\tProgress\tAccuracy\tLoss(Cross Entropy)" << endl;

        for(unsigned int epoch = 0; epoch < eras; ++epoch)
        {
            cout << epoch + 1 << "/" << eras << "\t";

            unsigned int batch_sequence[5] = {0, 1, 2, 3, 4};
            random_shuffle(&batch_sequence[0], &batch_sequence[4]);

            for(unsigned int i = 0; i < 5; ++i)
            {
                for(unsigned int j = batch_sequence[i] * batch_size; j < (batch_sequence[i] + 1) * batch_size; ++j)
                {
                    Tensor prediction = net.Forward(train_images[j]);
                    Tensor gradient = net.LossGradient(train_labels[j], prediction);
                    net.Backward(gradient);
                }
                cout << "--";

                
            }

            double loss = net.Loss(test_images, test_labels, 1000);
            double accur = net.Accuracy(test_images, test_labels, 1000);

            cout << "\t" << accur << '%' << "\t\t" << loss;
            cout << "\t\t" << (net.SaveModel("avg_" + to_string((int)accur) + '.' + to_string((int)(100 * accur) % 100)) ? "\tModel saved" : "\tFailed to save model...") << endl;
        }
    }

}


int main()
{

    TrainCNN_CIFAR10();

    return 0;
}