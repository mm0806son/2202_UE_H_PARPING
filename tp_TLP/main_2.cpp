#include <omp.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>

// Switching between different //ization strategies.

#define SERIAL 0
#define PARFOR 1
#define TASKS 2
#define TASKS_NAIVE 3

// ! Change here to choose strategy
// #define STRATEGY SERIAL
#define STRATEGY TASKS
// #define STRATEGY PARFOR

std::vector<double> compute_signal(size_t num_samples, double A, double fm, double fp, double tmin, double dt)
{
    std::vector<double> signal(num_samples, 100. * A);
#if STRATEGY == SERIAL
    for (size_t i = 0; i < signal.size(); ++i)
    {
        double t = tmin + i * dt;
        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
    }
#elif STRATEGY == TASKS
// TODO Use tasks (fixed or dynamic number) to compute the signal
#pragma omp parallel
    {
#pragma omp single
        {
            // for (size_t i = 0; i < signal.size(); ++i)
            {
#pragma omp task
                {
                    for (size_t i = 0; i < signal.size() / 3; ++i)
                    {
                        double t = tmin + i * dt;
                        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
                    }
                }

#pragma omp task
                {
                    for (size_t i = signal.size() / 2; i < signal.size() / 3 * 2; ++i)
                    {
                        double t = tmin + i * dt;
                        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
                    }
                }

#pragma omp task
                {
                    for (size_t i = signal.size() / 3 * 2; i < signal.size(); ++i)
                    {
                        double t = tmin + i * dt;
                        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
                    }
                }
            }
        }
    }

#elif STRATEGY == PARFOR
    // TODO Use a parfor to compute the signal
    omp_set_num_threads(4);
#pragma omp parallel for
    for (size_t i = 0; i < signal.size(); ++i)
    {
        double t = tmin + i * dt;
        signal[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
    }

#else
    std::cerr << "Not implemented!" << std::endl;
#endif
    return signal;
}

void plot_results(const std::vector<double> &samples, double tmin, double dt)
{
    FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplot_pipe, "set title 'Nice signal'\n");
    // fprintf(gnuplot_pipe, "set style data lp\n"); // Use to see markers
    fprintf(gnuplot_pipe, "set style data l\n");
    fprintf(gnuplot_pipe, "set xlabel 't'\n");
    fprintf(gnuplot_pipe, "set ylabel 's(t)'\n");
    // fprintf(gnuplot_pipe, "set term dumb\n"); // For fun
    fprintf(gnuplot_pipe, "plot '-'\n");
    for (size_t i = 0; i < samples.size(); ++i)
        fprintf(gnuplot_pipe, "%lf %lf\n", tmin + dt * i, samples[i]);
    fprintf(gnuplot_pipe, "e\n");
    fclose(gnuplot_pipe);
}

int main(int argc, char *argv[])
{
    /*
     * Check arguments and set things up
     */
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " N fm fp tmin tmax" << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_samples = std::stoull(argv[1]);
    double fm = std::stod(argv[2]), fp = std::stod(argv[3]), tmin = std::stod(argv[4]), tmax = std::stod(argv[5]), A = 0.5;
    double dt = std::abs(tmax - tmin) / num_samples;

    /*
     * Compute the signal and measure execution time
     */
    auto start = std::chrono::steady_clock::now();
    std::vector<double> signal = compute_signal(num_samples, A, fm, fp, tmin, dt);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Computed signal in: " << elapsed_seconds.count() << "s" << std::endl;

    /*
     * Plot the signal
     */
    plot_results(signal, tmin, dt);

    return EXIT_SUCCESS;
}
