#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <vector>
using namespace std;

#include "Net.cpp"
#include "Tensor.cpp"

void TrainMNIST()
{
    cout << "Train MNIST" << endl;;

    vector<Tensor> train_images;
    vector<unsigned int> train_labels;

    vector<Tensor> test_images;
    vector<unsigned int> test_labels;

    cout << "Data loading...";

    // Загрузка данных - - - - - - - - - - - - - - - - - - - - - -
    {
        unsigned int train_image_amount = 60000; // 60000
        unsigned int test_image_amount = 10000; // 10000

        string path = "data/";

        // Чтение тренировачных меток
        {
            ifstream inputFileStream(path + "train-labels", ios::in | std::ios::binary);    
            if (inputFileStream.fail()) throw std::runtime_error("Failed to open " + path + "train-labels");
            
            byte cursor;

            // Пропуск служебных байтов
            for(int i = 0; i < 8; ++i)
                inputFileStream.read((char*)&cursor, sizeof(cursor) );
            
            train_labels.reserve(train_image_amount);
            for(int i = 0; i < train_image_amount; ++i)
            {
                inputFileStream.read((char*)&cursor, sizeof(cursor) );
                train_labels.push_back((int)cursor);
            }

            inputFileStream.close();
        }

        // Чтение тренировачных изображений
        {
            ifstream inputFileStream(path + "train-images", ios::in | std::ios::binary);    
            if (inputFileStream.fail()) throw std::runtime_error("Failed to open " + path + "train-images");
            
            byte cursor;

            // Пропуск служебных байтов
            for(int i = 0; i < 16; ++i)
                inputFileStream.read((char*)&cursor, sizeof(cursor));
            
            train_images.reserve(train_image_amount);
            for(int image_index = 0; image_index < train_image_amount; ++image_index)
            {
                Tensor img = Tensor(1, 784, 1);

                for(int row = 0; row < 28; ++row)
                {
                    for(int col = 0; col < 28; ++col)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor));
                        img[0][row * 28 + col][0] = (double)cursor/255.0;
                    }
                }

                train_images.push_back(img);
            }

            inputFileStream.close();
        }

        // Чтение тестовых меток
        {
            ifstream inputFileStream(path + "test-labels", ios::in | std::ios::binary);    
            if (inputFileStream.fail()) throw std::runtime_error("Failed to open " + path + "test-labels");
            
            byte cursor;

            // Пропуск служебных байтов
            for(int i = 0; i < 8; ++i)
                inputFileStream.read((char*)&cursor, sizeof(cursor) );
            
            test_labels.reserve(test_image_amount);
            for(int i = 0; i < test_image_amount; ++i)
            {
                inputFileStream.read((char*)&cursor, sizeof(cursor) );
                test_labels.push_back((int)cursor);
            }

            inputFileStream.close();
        }

        // Чтение тестовых изображений
        {
            ifstream inputFileStream(path + "test-images", ios::in | std::ios::binary);    
            if (inputFileStream.fail()) throw std::runtime_error("Failed to open " + path + "test-images");
            
            byte cursor;

            // Пропуск служебных байтов
            for(int i = 0; i < 16; ++i)
                inputFileStream.read((char*)&cursor, sizeof(cursor));
            
            test_images.reserve(test_image_amount);
            for(int image_index = 0; image_index < test_image_amount; ++image_index)
            {
                Tensor img = Tensor(1, 784, 1);

                for(int row = 0; row < 28; ++row)
                {
                    for(int col = 0; col < 28; ++col)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor));
                        img[0][row * 28 + col][0] = (double)cursor/255.0;
                    }
                }

                test_images.push_back(img);
            }

            inputFileStream.close();
        }
    }
    
    cout << "done!" << endl;

    double learning_rate = 0.02;

    Net net = Net(learning_rate, 0, 0.5);

    cout << endl << "Accuracy on test images: " << net.Accuracy(test_images, test_labels, 1000) << '%' << endl;

    /*
        Обучать на всех данных, конечно, можно, но лучше разбить 
        данные на подвыборки и обучать сеть на них

        Да, я понимаю, что данный пример очень плох и долго обучается,
        но он создан только для изучения. Веса сети никуда не сохраняются
        
        При первых трёх запусках максимальная точность предсказаний достигла ~70% 
    */

    unsigned int eras = 50;
    unsigned int sample_size = 4000;

    cout << endl << "Epoch\tProgress\tAccuracy\tLoss(Cross Entropy)" << endl; 

    for(unsigned int epoch = 0; epoch < eras; ++epoch)
    {
        cout << epoch + 1 << "/" << eras << "\t";

        for(unsigned int i = 0; i < sample_size; ++i)
        {
            unsigned int idx = rand() % train_images.size();

            Tensor prediction = net.Forward(train_images[idx]);
            
            Tensor gradient = net.LossGradient(train_labels[idx], prediction);

            net.Backward(gradient);

            if (i % (sample_size / 10) == 0)
                cout << '.';
        }

        double accur = net.Accuracy(test_images, test_labels, 1000);
        double loss = net.Loss(test_images, test_labels, 1000);

        cout << "\t" << accur << '%';
        cout << "\t\t" << loss << endl;
        net.SaveModel(to_string(accur));

        if (learning_rate > 0.00005)
        {
            learning_rate *= 0.33;
            net.SetLearningRate(learning_rate);
        }
    }

    cout << endl << "Total accuracy on test images: " <<  net.Accuracy(test_images, test_labels) << '%';
    
}


int main()
{
    TrainMNIST();

    return 0;
}