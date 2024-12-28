#ifndef REGRESSION
#define REGRESSION

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class LinearRegression
{

private:

    double* W;
    double bias;
    int inputs;
    double learning_rate;

public:
    LinearRegression(int inputs, double learning_rate);

    double Predict(double* x);
    void Learn(double* x, double y);
    void Learn(double** X, double* Y, int dataset_size, int data_amount);

    double Loss(int amount, double** X, double* Y);
};


LinearRegression::LinearRegression(int inputs, double learning_rate = 0.0001)
{
    this->learning_rate = learning_rate;
    this->inputs = inputs;

    W = new double[inputs];
    for(int i = 0; i < inputs; ++i)
        W[i] = 0;
    
    bias = 0;
}

double LinearRegression::Predict(double* x)
{
    double ans = bias;

    for(int i = 0; i < inputs; ++i)
        ans += W[i] * x[i];

    return ans;
}

void LinearRegression::Learn(double* x, double y)
{
    double prediction = Predict(x);

    // double e = (y - prediction) > 0 ? 1 : -1; // Вот это производная MAE, но с линейной производной получается лучше
    double e = (y - prediction);
    double e_with_lr = e * learning_rate; // Только для ускорения вычисления

    for(int i = 0; i < inputs; ++i)
    {
        double dw = e_with_lr * x[i];
        W[i] = W[i] + dw;
    }
    bias += e_with_lr;
}

#include <random>
void LinearRegression::Learn(double** X, double* Y, int dataset_size, int data_amount = 0)
{

    if (data_amount <= 0)
    {
        for(int i = 0; i < dataset_size; ++i)
            Learn(X[i], Y[i]);
    }
    else
    {
        for(int i = 0; i < data_amount; ++i)
        {
            int randi = rand() % dataset_size;
            Learn(X[randi], Y[randi]);
        }
    }

    
}

// MAE Loss 
double LinearRegression::Loss(int amount, double** X, double* Y) 
{
    double SE = 0;

    for(int i = 0; i < amount; ++i)
    {
        double predicion = Predict(X[i]);
        SE += abs(Y[i] - predicion);
    }

    return SE / amount;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#endif