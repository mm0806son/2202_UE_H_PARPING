#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <complex>
#include <cmath>
#include <algorithm>

#include <mpi.h>

using namespace std::complex_literals;

std::vector<double> compute_signal(size_t num_samples, double A, double fm, double fp, double dt, double tmin)
{
    std::cout << "Node 0: sampling signal" << std::endl;
    std::vector<double> samples(num_samples, 10. * A);

    // Compute the signal
    for(size_t i = 0; i < samples.size(); ++i)
    {
        double t = tmin + i * dt;
        samples[i] = (1.0 + A * std::cos(2. * M_PI * fm * t)) * std::cos(2. * M_PI * fp * t);
    }

    return samples;
}

std::vector<std::complex<double>> compute_dft_rows(size_t num_samples, size_t first_row, size_t last_row)
{
    if(num_samples == 0) return {};
    size_t number_rows = last_row - first_row + 1;
    // Compute the DFT matrix rows
    double N = num_samples;
    std::vector<std::complex<double>> dft_matrix(num_samples * number_rows, 0.0);
    std::complex<double> w = std::exp(-2.0i * M_PI / N);
    for(size_t i = 0; i < num_samples * number_rows; ++i)
        dft_matrix[i] = std::pow(w, (first_row + i % num_samples) * (first_row + i / num_samples)) / std::sqrt(N);

    return dft_matrix;
}

std::vector<std::complex<double>> compute_partial_dft_of_signal(const std::vector<std::complex<double>>& dft_rows, const std::vector<double>& signal)
{
    if(signal.size() == 0) return {};
    size_t number_rows = dft_rows.size() / signal.size(), num_samples = signal.size();
    std::vector<std::complex<double>> dft_signal(number_rows, 0.0);
    // Compute the relevant part of the DFT
    for(size_t i = 0; i < num_samples * number_rows; ++i)
        dft_signal[i / num_samples] += dft_rows[i] * signal[i % num_samples] * std::pow(-1.+0i, i % num_samples);
    // THe std::pow(-1.+0.*1i, i % num_samples) is here to avoid doing the fftshift after computing the DFT

    return dft_signal;
}

void plot_results(const std::vector<double>& samples, const std::vector<double>& dft,
                  double tmin, double dt)
{
    FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplot_pipe, "set title 'Nice signal'\n");
    // fprintf(gnuplot_pipe, "set style data lp\n"); // Use to see markers
    fprintf(gnuplot_pipe, "set style data l\n");
    fprintf(gnuplot_pipe, "set xlabel 't'\n");
    fprintf(gnuplot_pipe, "set ylabel 's(t)'\n");
    // fprintf(gnuplot_pipe, "set term dumb\n"); // For fun
    fprintf(gnuplot_pipe, "plot '-'\n");
    for(size_t i = 0; i < samples.size(); ++i)
        fprintf(gnuplot_pipe, "%lf %lf\n", tmin + dt * i, samples[i]);
    fprintf(gnuplot_pipe, "e\n");
    fclose(gnuplot_pipe);

    FILE *gnuplot_pipe_fft = popen("gnuplot -persistent", "w");
    fprintf(gnuplot_pipe_fft, "set title 'Spectrum of the nice signal'\n");
    // fprintf(gnuplot_pipe_fft, "set style data lp\n"); // Use to see markers
    fprintf(gnuplot_pipe_fft, "set style data l\n");
    fprintf(gnuplot_pipe_fft, "set xlabel 'f'\n");
    fprintf(gnuplot_pipe_fft, "set ylabel 'A(f)'\n");
    // fprintf(gnuplot_pipe_fft, "set term dumb\n"); // For fun
    fprintf(gnuplot_pipe_fft, "plot '-'\n");
    double df = 1.0 / dt / samples.size(), fmax = 1.0 / dt;
    for(size_t i = 0; i < dft.size(); ++i)
        fprintf(gnuplot_pipe_fft, "%lf %lf\n", df * i - fmax / 2. , dft[i]);
    fprintf(gnuplot_pipe_fft, "e\n");
    fclose(gnuplot_pipe_fft);
}

int dispatch_row_ranges(size_t num_samples, int number_compute_nodes)
{
    // TODO Send to the nodes the min and max indices of the rows they will have to compute
    // be carefull, all nodes might not have the same workload
    return 0;
}

int send_signal(const std::vector<double>& samples, int number_compute_nodes)
{
    // TODO send the signal to each node
    return 0;
}

int receive_dft(std::vector<double>& dft, int number_compute_nodes)
{
    // TODO receive the partial DFTs from the nodes and put them into the correct part of dft
    // be carefull, all nodes might not have the same workload
    return 0;
}

int receive_row_ranges(size_t& num_samples, size_t& first_row, size_t& last_row)
{
    // TODO receive the min and max indices of the rows to compute
    return 0;
}

int receive_signal(std::vector<double>& samples)
{
    // TODO receive the signal from master
    return 0;
}

int send_back_partial_dft(const std::vector<double>& abs_dft_signal)
{
    // TODO send back the partial DFTs to master
    // be carefull, all nodes might not have the same workload
    return 0;
}

int main(int argc, char* argv[])
{
    /*
     * Check arguments and set things up
     */
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " N fm fp tmin tmax" << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_samples = std::stoull(argv[1]);
    double fm = std::stod(argv[2]), fp = std::stod(argv[3]), tmin = std::stod(argv[4]), tmax = std::stod(argv[5]), A = 0.5;
    double dt = std::abs(tmax - tmin) / num_samples;

    /*
     * Set up the MPI environment
     */
    int err, total_number_nodes, id;
    err = MPI_Init(&argc, &argv);
    if(err != MPI_SUCCESS)
    {
        std::cerr << "Failed MPI init" << std::endl;
        return EXIT_FAILURE;
    }

    err = MPI_Comm_size(MPI_COMM_WORLD, &total_number_nodes);
    if(err != MPI_SUCCESS)
    {
        std::cerr << "Failed MPI call" << std::endl;
        return EXIT_FAILURE;
    }
    err = MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if(err != MPI_SUCCESS)
    {
        std::cerr << "Failed MPI call" << std::endl;
        return EXIT_FAILURE;
    }

    size_t number_compute_nodes = total_number_nodes - 1;
    /*
     * Main work code
     */
    if(id == 0)
    {
        // if we are on the master node
        std::cout << "Message from master:\n\tWorld size : " << total_number_nodes << std::endl;

        // To get the DFT of the signal we need to compute a num_samples * num_samples matrix
        // and multiply it to the signal vector
        // A trivial way to do this is to compute different rows of the matrix on different nodes
        // and apply them to the signal
        // First we split the rows between the compute nodes:
        err = dispatch_row_ranges(num_samples, number_compute_nodes);

        // Now that the nodes are working, let's compute our signal
        std::vector<double> samples = compute_signal(num_samples, A,  fm,  fp,  dt,  tmin);

        // Now we need to send the signal to the node to compute the matrix vector product
        err = send_signal(samples, number_compute_nodes);

        // Wait for the results and reassemble the partial DFTs!
        std::vector<double> dft(num_samples, 0.0);
        err = receive_dft(dft, number_compute_nodes);

        plot_results(samples, dft, tmin, dt);
    }
    
    if(id != 0)
    {
        size_t num_samples = 0, first_row = 0, last_row = 0;

        // Get our orders
        err = receive_row_ranges(num_samples, first_row, last_row);
        std::cout << "Node " << id << " received orders to compute rows " << first_row 
                  << " to " << last_row << std::endl;

        // Compute the DFT while master gets the signal
        std::cout << "Node " << id << " computes the DFT matrix rows" << std::endl;
        std::vector<std::complex<double>> dft_rows = compute_dft_rows(num_samples, first_row, last_row);

        // Get the signal
        std::vector<double> samples(num_samples, 0.0);
        err = receive_signal(samples);
        std::cout << "Node " << id << " received the signal" << std::endl;

        // Compute the partial DFT of the signal
        std::vector<std::complex<double>> dft_signal = compute_partial_dft_of_signal(dft_rows, samples);

        // We only send back the absolute value of the DFT coefs for convenience
        std::vector<double> abs_dft_signal(last_row - first_row + 1, 0.0);
        std::transform(dft_signal.cbegin(), dft_signal.cend(), abs_dft_signal.begin(),
                       [](std::complex<double> v) {return std::abs(v);});

        std::cout << "Node " << id << " is sending back its dft part" << std::endl;
        err = send_back_partial_dft(abs_dft_signal);
    }

    /*
     * Close up shop
     */
    err = MPI_Finalize();
    if(err != MPI_SUCCESS)
    {
        std::cerr << "Failed MPI finalize" << std::endl;
        return EXIT_FAILURE;
    }
        std::cout << "Node " << id << " going on vacation." << std::endl;

    return EXIT_SUCCESS;
}
