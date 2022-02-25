#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <cassert>

#include <mpi.h>
#include <openssl/sha.h>

std::vector<std::string> read_dict(const std::filesystem::path& path, size_t words_to_read)
{
    std::ifstream if_dict(path);
    std::vector<std::string> words(words_to_read);
    size_t w;

    for(w = 0; w < words_to_read; ++w)
    {
        if(!if_dict.good())
            break;
        std::getline(if_dict, words[w]);
    }
    if(w < words_to_read)
    {
        words.resize(w - 1);
        std::cerr << "The dictionary contained less words than requested." << std::endl;
    }
    std::cout << "Loaded " << w  << " words." << std::endl;

    return words;
}

// Code recovered from stackoverflow #2262386
std::string sha512(const std::string& str)
{
    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512_CTX sha512;
    SHA512_Init(&sha512);
    SHA512_Update(&sha512, str.c_str(), str.size());
    SHA512_Final(hash, &sha512);
    std::stringstream ss;
    for(int i = 0; i < SHA512_DIGEST_LENGTH; i++)
    {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

std::string pack_data(const std::vector<std::string>& words, size_t start_index, size_t last_index)
{
    std::stringstream sstream;
    for(size_t i = start_index; i < last_index; ++i)
        sstream << words[i] << " ";
    sstream << words[last_index];
    return sstream.str(); // Note that this is slow as it copies the buffer, but that's good enough for this exercise
}

std::string look_for_hash(const std::string& hash_to_look_for, const std::vector<std::string>& words, size_t& hash_count)
{
    hash_count = 0;

    for(const std::string& word: words)
    {
        ++hash_count;
        if(sha512(word) == hash_to_look_for)
            return word;
    }
    return {};
}

char* allocate_char_buffer(size_t size)
{
    // The correct way of allocating in cpp is to use unique_ptr but we make it look
    // more like C/Java for people not familiar with c++
    // Do no forget to call free_char_buffer!
    // std::unique_ptr<char[]> data = std::make_unique<char[]>(num_bytes_to_receive);
    return new char[size];
}

void free_char_buffer(char* buffer)
{
    delete[] buffer;
    buffer = nullptr;
}

std::vector<std::string> string_buffer_to_string_vector(const char* buffer, size_t buffer_size)
{
    std::stringstream sstream;
    std::vector<std::string> words;
    std::string word;
    sstream.write(buffer, buffer_size);
    while(sstream.good())
    {
        words.emplace_back();
        sstream >> words.back();
    }
    return words;
}

/*
 * You can use these tags for your MPI messages
 */
enum MPI_tags {
    WORD,
    LAST_WORD,
    RESULT
};

int send_words_to_nodes(const std::vector<std::string>& words, int number_nodes)
{
    /*
     * TODO You must send the words to the cluster computer nodes There are many
     * alternatives more or less efficient.  For instance, you can try to send
     * them word by word.
     * HINT : The std::string can be used as the buffer for its own MPI_SEND
     */
    for(const std::string& word: words)
    {
    }

    return 0;
}

int receive_words(std::vector<std::string>& words)
{
    /*
     * TODO Now you need to receive the words on each compute node.  The
     * structure of this function will depend on how you choose to send the
     * words Keep in mind that you must have a way to tell when you have
     * received everything.
     *
     * You can allocate the receive buffers using allocate_char_buffer()
     * and free them using free_char_buffer().
     */
    MPI_Status status;
    int num_bytes_to_receive;
    char* buffer = allocate_char_buffer(num_bytes_to_receive);
    free_char_buffer(buffer);

    return 0;
}

int send_back_word(const std::string& word)
{
    /*
     * TODO: in this function you must send back to master the word you have found
     * or an empty string (meaning a string containing only the terminating character)
     * to the master node.
     */
    return 0;
}

int receive_found_word(std::string& word, int number_nodes)
{
    /*
     * TODO: Now you must wait for the results of the nodes and see if one of them found the
     * word you provided the hash of.
     */
    for(int node_id = 1; node_id <= number_nodes; ++node_id)
    {
        MPI_Status status;
        int num_bytes_to_receive;
        char* buffer = allocate_char_buffer(num_bytes_to_receive);

        word = std::string(buffer);
        free_char_buffer(buffer);
    }

    std::cout << "Nobody found the word :'(" << std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
    int id, err, total_number_nodes;

    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " dictionary_filename number_words_to_load hash_to_look_for" << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path dict_path(argv[1]);
    size_t n_words = std::stoull(argv[2]);
    if(!std::filesystem::is_regular_file(std::filesystem::status(dict_path)))
    {
        std::cerr << "Invalid dictionary file." << std::endl;
        return EXIT_FAILURE;
    }

    /*
     * Set things up
     */
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

    // The arguments are available to everybody
    std::string hash_to_look_for(argv[3]);

    /*
     * Main work code
     */
    if(id == 0)
    {
        // if we are on the master node
        size_t number_nodes = total_number_nodes - 1;
        std::cout << "Message from master:\n\tWorld size : " << total_number_nodes << std::endl;

        // Load the dictionary
        std::vector<std::string> words = read_dict(dict_path, n_words);

        err = send_words_to_nodes(words, number_nodes);

        std::string word;
        err = receive_found_word(word, number_nodes);
    }

    if(id != 0)
    {
        // Receive the word buffer
        std::vector<std::string> words;
        err = receive_words(words);

        // Look for the hash
        size_t num_hashes_computed = 0;
        std::string word = look_for_hash(hash_to_look_for, words, num_hashes_computed);
        std::cout << "Node " << id << " computed " << num_hashes_computed << " hases." << std::endl;

        // Send back the word to master (send empty word if we found nothing)
        err = send_back_word(word);
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
