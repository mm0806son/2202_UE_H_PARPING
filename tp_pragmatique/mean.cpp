#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <numeric>
#include <chrono>

int main(int argc, char **argv) {
  using data_type = double;
  std::ifstream fd(argv[1]);
  std::string buffer;
  std::vector<data_type> data;

  // data extraction part
  std::cout << "> reading " << argv[1] << std::endl;

  auto start = std::chrono::system_clock::now();
  while(fd) {
    std::getline(fd, buffer);
    if(fd)
      data.push_back(std::stoul(buffer));
  }
  auto mid = std::chrono::system_clock::now();

  std::cout << "done in " << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count() << " ms" << std::endl;

  // computation part
  std::cout << "> processing data " << std::endl;
  data_type mean = std::accumulate(data.begin(), data.end(), data_type(0)) / data.size();
#if I_FEEL_CXXISH
  data_type absdiff = std::accumulate(data.begin(), data.end(), data_type(0),
                                      [mean](data_type acc, data_type elt) {
                                        return acc + ((elt > mean)?(elt - mean): (mean-elt));
                                      });
#else
  data_type absdiff =0;
  for(size_t i = 0; i < data.size(); ++i) {
    data_type elt = data[i];
    absdiff += (elt > mean)?(elt - mean): (mean-elt);
  }
#endif
  data_type mean_absdiff = absdiff / data.size();
  auto stop = std::chrono::system_clock::now();
  std::cout << "done in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - mid).count() << " ms" << std::endl;

  std::cout << "> mean of absolute diff to the mean" << std::endl << mean_absdiff << std::endl;
  return 0;
}
