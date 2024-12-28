#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
using namespace std;

#include "LinearRegression.cpp"

void LinearFunctionPrediction()
{
    cout << endl << "> Prediction of linear function y = 2x" << endl << endl;

    // Создание данных для обучения - - - - - - - - - - - - -
    // Предсказание y = 2x

    int n = 100;

    double** X = new double*[n];
    double* Y_learn = new double[n];
    double* Y_test = new double[n];

    // Создание генератора шума, распределенного нормально (генератор остатков)
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution d = std::normal_distribution(0.0, 10.0);
    auto random_double = [&d, &gen]{ return d(gen); };

    for(int i = 0; i < n; ++i)
    {
        X[i] = new double[1]{(double)(rand() % 100 - 50)};
        Y_test[i] = X[i][0] * 2;
        Y_learn[i] = X[i][0] * 2 + random_double();
    }

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - -

    LinearRegression r(1, 0.0001);

    int epoch_amount = 5;

    for(int epoch = 0; epoch < epoch_amount; ++epoch)
    {
        cout << "\tEpoch: " << epoch + 1 << endl;
        cout << "\tLoss (MAE): " << r.Loss(n, X, Y_test) << endl;
        cout << endl;

        r.Learn(X, Y_learn, n, 30);
    }
}

void FoodDeliveryTimePrediction()
{
    cout << endl << "> Prediction of food delivery time" << endl << endl;

    int X_row_length = 7;

    int learn_dataset_size = 600;
    double** X_learn = new double*[learn_dataset_size];
    double* Y_learn = new double[learn_dataset_size];

    int test_dataset_size = 283;
    double** X_test = new double*[test_dataset_size];
    double* Y_test = new double[test_dataset_size];

    // Парсим и загружаем датасет

    ifstream file; 
    file.open("Food_Delivery_Times_for_regression.csv");
    
    if ( file.is_open() ) 
    {
        // Пропуск строки с названиями колонок
        string line;
        getline (file, line);

        // Первая колонка - индексы заказов, которые не несут смысла
        double Order_id, Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time_min, Courier_Experience_yrs, Delivery_Time_min;
        int row_count = 0;

        while(file >> Order_id >> Distance_km >> Weather >> Traffic_Level >> Time_of_Day >> Vehicle_Type >> Preparation_Time_min >> Courier_Experience_yrs >> Delivery_Time_min)
        {
            if (row_count < learn_dataset_size)
            {
                X_learn[row_count] = new double[X_row_length]{Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time_min, Courier_Experience_yrs};
                Y_learn[row_count] = Delivery_Time_min;
            }
            else
            {
                X_test[row_count - learn_dataset_size] = new double[X_row_length]{Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time_min, Courier_Experience_yrs};
                Y_test[row_count - learn_dataset_size] = Delivery_Time_min;
            }

            ++row_count;
        }
    }

    // Обучение

    LinearRegression r(X_row_length, 1.0E-5);

    int epoch_amount = 200;

    for(int epoch = 0; epoch < epoch_amount; ++epoch)
        r.Learn(X_learn, Y_learn, learn_dataset_size / 2);

    cout << "\tLoss (MAE): " << r.Loss(test_dataset_size, X_test, Y_test) << endl;

    cout << endl << "\tTests:" << endl;
    cout << endl << "\tExpect\tPredict\tDelta" << endl;

    for(int i = 0; i < 10; ++i)
    {
        int k = rand() % test_dataset_size;
        int prediction = r.Predict(X_test[k]);

        cout << "\t  " << Y_test[k] << "\t  ";
        cout << prediction << "\t  ";
        cout << abs(Y_test[k] - prediction) << endl;
    }
    cout << endl;

}

int main()
{
    LinearFunctionPrediction();
    FoodDeliveryTimePrediction();

    return 0;
}