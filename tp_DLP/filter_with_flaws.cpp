// Give the input and output file names on the command line
// compile: g++ -o test filter.cpp -lsndfile

#include <sndfile.h>

#include <algorithm>
#include <array>
#include <iostream>

#ifdef WITH_XSIMD
#include "xsimd/xsimd.hpp"
#endif

using namespace std;

//////////////////////////////////////////////////////////////
//  Filter Code Definitions
//////////////////////////////////////////////////////////////

// maximum number of inputs that can be handled
// in one function call
constexpr size_t MAX_INPUT_LEN = 80;
// maximum length of filter than can be handled
constexpr size_t MAX_FLT_LEN = 63;
// buffer to hold all of the input samples
constexpr size_t BUFFER_LEN = (MAX_FLT_LEN - 1 + MAX_INPUT_LEN);

// FIR init
template <class T, size_t N>
void firFixedInit(std::array<T, N> &insamp)
{
    std::fill(std::begin(insamp), std::end(insamp), T(0));
}

// the FIR filter function
template <class T, size_t N, size_t M, size_t P>
void firFixed(std::array<T, P> &insamp,
              std::array<int16_t, N> const &coeffs,
              std::array<int16_t, M> const &input,
              std::array<int16_t, M> &output, int length)
{
    // put the new samples at the high end of the buffer
    std::copy(input.begin(), input.begin() + length, &insamp[coeffs.size() - 1]);

    // apply the filter to each input sample
#ifdef WITH_XSIMD
    static_assert(std::is_same<T, int32_t>::value, "compatible types");
    constexpr int batch_size = xsimd::batch<int32_t>::size;

    for (int n = 0; n < length; n++)
    {
        // calculate output n

        // load rounding constant
        xsimd::batch<int32_t> acc(1 << 14); // accumulator for MACs
        // perform the multiply-accumulate
        for (size_t k = 0; k < coeffs.size(); k++)
        {
            acc += xsimd::batch<int32_t>(coeffs[k]) * xsimd::load_unaligned(&insamp[coeffs.size() - 1 + n - k]);
        }
        // saturate the result
        acc = xsimd::min(acc, 0x3fffffff);
        acc = xsimd::max(acc, -0x40000000);

        // convert from Q30 to Q15
        int32_t tmp[batch_size];
        (acc >> 15).store_unaligned(&tmp[0]);
        std::copy(std::begin(tmp), std::end(tmp), &output[n]);
    }

#else
    for (int n = 0; n < length; n++)
    {
        // calculate output n

        // load rounding constant
        int32_t acc = 1 << 14; // accumulator for MACs
        // perform the multiply-accumulate
        for (size_t k = 0; k < coeffs.size(); k++)
        {
            acc += (int32_t)coeffs[k] * (int32_t)insamp[coeffs.size() - 1 + n - k];
        }
        // saturate the result
        if (acc > 0x3fffffff)
        {
            acc = 0x3fffffff;
        }
        else if (acc < -0x40000000)
        {
            acc = -0x40000000;
        }
        // convert from Q30 to Q15
        output[n] = (int16_t)(acc >> 15);
    }
#endif

    // shift input samples back in time for next time
    std::copy(&insamp[length], &insamp[length + coeffs.size() - 1], &insamp[0]);
}

// filter obtained with octave (load pkg signal)
// octave command lines:
// c=round(fir1(62, .05)/sqrt(sum(fir1(62, .05).^2))*2^13)
// for i=1:63 printf("%d, ", c(i)) endfor ; printf("\n")
constexpr size_t FILTER_LEN = 63;
std::array<int16_t, FILTER_LEN> coeffs = {
    -35, -37, -41, -45, -50, -55, -58, -57, -52, -39, -18,
    14, 58, 116, 189, 276, 379, 495, 625, 766, 915, 1069,
    1225, 1379, 1528, 1666, 1791, 1899, 1986, 2051, 2090, 2103, 2090,
    2051, 1986, 1899, 1791, 1666, 1528, 1379, 1225, 1069, 915, 766,
    625, 495, 379, 276, 189, 116, 58, 14, -18, -39, -52,
    -57, -58, -55, -50, -45, -41, -37, -35};

// number of samples to read per loop
constexpr size_t SAMPLES = 80;

int main(int argc, char *argv[])
{

    // open the input waveform file

    if (argc != 3) /* argc should be 2 for correct execution */
    {
        std::cout << "usage: " << argv[0]
                  << "input_pcm_filename output_pcm_filename\n";
        return 0;
    }

    SNDFILE *infile;
    SNDFILE *outfile;
    sf_count_t count;
    SF_INFO sfinfo;
    if ((infile = sf_open(argv[1], SFM_READ, &sfinfo)) == NULL)
    {
        std::cerr << "Error: Not able to open input file '" << argv[1] << "'\n";
        sf_close(infile);
        exit(1);
    }

    if ((outfile = sf_open(argv[2], SFM_WRITE, &sfinfo)) == NULL)
    {
        std::cerr << "Error: Not able to open output file '%" << argv[2] << "'\n";
        sf_close(infile);
        exit(1);
    }

    // array to hold input samples
#ifdef WITH_XSIMD
    std::array<int32_t, BUFFER_LEN> insamp;
#else
    std::array<int16_t, BUFFER_LEN> insamp;
#endif

    // initialize the filter
    firFixedInit(insamp);

    // run filter on successive batches of tthe buffer length until the input file
    // is empty
    std::array<int16_t, SAMPLES> input;
    std::array<int16_t, SAMPLES> output;
    while ((count = sf_read_short(infile, input.data(), input.size())) > 0)
    {
        firFixed(insamp, coeffs, input, output, count);
        sf_write_short(outfile, output.data(), count);
    }

    sf_close(infile);
    sf_close(outfile);
    return 0;
}
