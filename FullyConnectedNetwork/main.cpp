#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <vector>
using namespace std;

#include "Matrix.cpp"
#include "SimpleNet.cpp"


void PredictNumbersMNIST()
{
    vector<Matrix> train_images;
    vector<Matrix> train_labels;

    vector<Matrix> test_images;
    vector<Matrix> test_labels;

    // Загрузка данных - - - - - - - - - - - - - - - - - - - - - -
    {
        string path = "data/";

        // Чтение тренировачных меток
        {
            ifstream inputFileStream(path + "train-labels", ios::in | std::ios::binary);    
            if (inputFileStream.fail()) throw std::runtime_error("Failed to open " + path + "train-labels");
            
            byte cursor;

            // Пропуск служебных байтов
            for(int i = 0; i < 8; ++i)
                inputFileStream.read((char*)&cursor, sizeof(cursor) );
            
            train_labels.reserve(60000);
            for(int i = 0; i < 60000; ++i)
            {
                inputFileStream.read((char*)&cursor, sizeof(cursor) );

                Matrix Y = Matrix(10, 1);
                Y[(int)cursor][0] = 1;
                train_labels.push_back(Y);
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
            
            train_images.reserve(60000);
            for(int image_index = 0; image_index < 60000; ++image_index)
            {
                Matrix img = Matrix(784, 1);

                for(int row = 0; row < 28; ++row)
                {
                    for(int col = 0; col < 28; ++col)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor));
                        img[row * 28 + col][0] = (double)cursor/255.0;
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
            
            test_labels.reserve(10000);
            for(int i = 0; i < 10000; ++i)
            {
                inputFileStream.read((char*)&cursor, sizeof(cursor) );

                Matrix Y = Matrix(10, 1);
                Y[(int)cursor][0] = 1;
                test_labels.push_back(Y);
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
            
            test_images.reserve(10000);
            for(int image_index = 0; image_index < 10000; ++image_index)
            {
                Matrix img = Matrix(784, 1);

                for(int row = 0; row < 28; ++row)
                {
                    for(int col = 0; col < 28; ++col)
                    {
                        inputFileStream.read((char*)&cursor, sizeof(cursor));
                        img[row * 28 + col][0] = (double)cursor/255.0;
                    }
                }

                test_images.push_back(img);
            }

            inputFileStream.close();
        }
    }
    
    SimpleNet net = SimpleNet(0.1);
    cout << endl << "Accuracy on test images: " << net.Accuracy(test_images, test_labels, 100) << '%' << endl;

    /*
        Обучать на всех данных, конечно, можно, но лучше разбить 
        данные на подвыборки и обучать сеть на них

        Да, я понимаю, что данный пример очень плох и долго обучается,
        но он создан только для изучения. Веса сети никуда не сохраняются
        
        При первых трёх запусках максимальная точность предсказаний достигла ~70% 
    */

    unsigned int eras = 30;
    unsigned int sample_size = 4000;

    cout << endl << "Epoch\tProgress\tAccuracy\tLoss(Cross Entropy)" << endl; 

    for(unsigned int epoch = 0; epoch < eras; ++epoch)
    {
        cout << epoch + 1 << "/" << eras << "\t";

        for(unsigned int i = 0; i < sample_size; ++i)
        {
            unsigned int idx = rand() % 60000;

            Matrix prediction = net.Forward(train_images[idx]);
            Matrix gradient = net.LossGradient(train_labels[idx], prediction);
            net.Backward(gradient);

            if (i % (sample_size / 10) == 0)
                cout << '.';
        }

        cout << "\t" << net.Accuracy(test_images, test_labels, 1000) << '%';
        cout << "\t\t" << net.Loss(test_images, test_labels, 100) << endl;
    }

    cout << endl << "Total accuracy on test images: " <<  net.Accuracy(test_images, test_labels) << '%';
    
}


int main()
{
    PredictNumbersMNIST();

    return 0;
}