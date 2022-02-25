#include <iostream>
#include <chrono>

#include <mpi.h>

int main(int argc, char* argv[])
{
    int id, err, size;
    double wtime;
    
    /*
     * Set things up
     */
    err = MPI_Init(&argc, &argv);
    if(err != MPI_SUCCESS)
    {
        std::cerr << "Failed MPI init" << std::endl;
        return EXIT_FAILURE;
    }

    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
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

    /*
     * Main work code
     */
    if(id == 0)
    {
        // if we are on the master node
        std::cout << "Message from master:\n\tWorld size : " << size << std::endl;
    }
    else
    {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        std::time_t date_now = std::time(nullptr);
        std::cout << "Message from compute node #" << id << ":\n\tms since epoch " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
        std::cout << "\tProcessor name: " << processor_name << std::endl;
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

    return EXIT_SUCCESS;
}
