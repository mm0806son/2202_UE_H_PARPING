#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <numeric>
#include <chrono>
#include <omp.h>

#define n_data 10 // 000000

using data_type = double;

std::vector<data_type> read_input(std::ifstream &fd)
{
  std::vector<data_type> data;
  std::string buffer;
#pragma omp parallel
  {
#pragma omp single
    {
      {
#pragma omp task
        {

          auto start = std::chrono::system_clock::now();

          for (int i = 0; i < n_data / 2; i++)
          {
            std::getline(fd, buffer);
            data.push_back(std::stod(buffer));
            std::cout << "thread1: " << buffer << std::endl;
          }

          auto mid = std::chrono::system_clock::now();
          std::cout << "getline done in " << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count() << " ms" << std::endl;

          // auto stqrt = std::chrono::system_clock::now();
          // for (int i = 0; i < n_data / 2; i++)
          // {

          //   data.push_back(std::stod(buffer));
          // }
          // auto end = std::chrono::system_clock::now();
          // std::cout << "push_back done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - stqrt).count() << " ms" << std::endl;
        }

#pragma omp task
        {

          // for (int i = 0; i < n_data / 2; i++)
          // {
          //   std::getline(fd, buffer);
          // }

          for (int i = n_data / 2; i < n_data; i++)
          {
            std::getline(fd, buffer);
            data.push_back(std::stod(buffer));
            std::cout << "thread2: " << buffer << std::endl;
          }
          std::cout << "test threqd 2" << std::endl;
          std::cout << data.size() << std::endl;
        }
      }
    }
  }
  return data;
}

int main(int argc, char **argv)
{

  std::ifstream fd(argv[1]);
  std::vector<data_type> data;

  // data extraction part
  std::cout << "> reading " << argv[1] << std::endl;

  auto start = std::chrono::system_clock::now();
  data = read_input(fd);
  auto mid = std::chrono::system_clock::now();

  for (size_t i = 0; i < data.size(); ++i)
  {
    std::cout << data[i] << ", ";
  }

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count() << " ms" << std::endl;

  // computation part
  std::cout << "> processing data " << std::endl;
  data_type mean = std::accumulate(data.begin(), data.end(), data_type(0)) / data.size();

// #define I_FEEL_CXXISH 0
#if I_FEEL_CXXISH
  data_type absdiff = std::accumulate(data.begin(), data.end(), data_type(0),
                                      [mean](data_type acc, data_type elt)
                                      {
                                        return acc + ((elt > mean) ? (elt - mean) : (mean - elt));
                                      });
#else
  data_type absdiff = 0;
  for (size_t i = 0; i < data.size(); ++i)
  {
    data_type elt = data[i];
    absdiff += (elt > mean) ? (elt - mean) : (mean - elt);
  }
#endif
  data_type mean_absdiff = absdiff / data.size();
  auto stop = std::chrono::system_clock::now();
  std::cout << "done in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - mid).count() << " ms" << std::endl;

  std::cout << "> mean of absolute diff to the mean" << std::endl
            << mean_absdiff << std::endl;
  return 0;
}
