#include <iostream>
#include <ctime>
#include <fstream>
#include "stdlib.h"
#define random(a, b) (rand() / (b - a) + a) // set range

using namespace std;

int main()
{
    ofstream out;
    out.open("data.txt", ios::app);

    srand((int)time(0)); // seed
    for (int i = 0; i < 10000000; i++)
    {
        float random_n = (rand() % (10000 - 1) + 1) + rand() / double(RAND_MAX);
        out << random_n << std::endl;
    }
    // out << "\n";
    out.close();
    return 0;
}
