#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>


#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


// added these
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
        -> wget -r <link for sift 1M>
        -> cd to sift tar, then unzip tar file
        -> move directory to outer level and rename to sift1M
 *
 * and unzip it to the sudirectory sift1M.
 **/



 /*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

/*******************************************************
 * Added for debugging
 *******************************************************/
const int debugFlag = 1;

void debugTime() {
	if (debugFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm *tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
		std::cout << timestamp << std::flush;
    }
}

//needs atleast 2 args always
//  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "--" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)



double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*******************************************************
 * Run tests
 *******************************************************/

//  args are nb, M, gamma
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "====================\nSTART: running MAKE_INDICES for hnsw --" << nthreads << "cores\n" << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw --...\n");
    double t0 = elapsed();
    
    int efc = 40; // default is 40
    int efs = 16; //  default is 16
    int k = 10; // search parameter
    size_t d = 128; // dimension of the vectors to index
    int M = 32; // HSNW param M
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    srand(0); // seed for random number generator


    // create sizes of 1k, 10k, 100k
    // size_t nb = 1000 * 10; // size of the database we plan to index
    size_t N = 0; // N will be how many we truncate nb from sift1M to
    
    int opt;
    {// parse arguments

        if (argc != 3) {
            fprintf(stderr, "Syntax: %s <number vecs>\n", argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        debug("N: %ld\n", N);

        // M = atoi(argv[2]);
        // debug("M: %d\n", M);

        gamma = atoi(argv[2]);
        debug("gamma: %d\n", gamma);
    }


    // generate metadata
    std::vector<int> metadata(N);
    for (int i = 0; i < N; i++) {
        int rand_num = rand() % gamma;
        metadata[i] = rand_num;
    }
    assert(N == metadata.size());

    // print metadata
    printf("[%.3f s] Set metadata, %ld attr's generated\n", 
        elapsed() - t0, metadata.size());
    // printf("Metadata: ");
    // for (int i = 0; i < nb; i++) {
    //     printf("%d, ", metadata[i]);
    // }
    // printf("\n");
    

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
               elapsed() - t0, d, M, N, gamma);
    // base index
    faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
    base_index.hnsw.efConstruction = efc; // default is 40 * gamma in HNSW.capp
    base_index.hnsw.efSearch = efs; // default is 16 * gamma in HNSW.capp
    // hybrid index
    faiss::IndexHNSWHybrid hybrid_index(d, M, gamma, metadata);
    hybrid_index.hnsw.efConstruction = efc; // default is 40 * gamma in HNSW.capp
    hybrid_index.hnsw.efSearch = efs; // default is 16 * gamma in HNSW.capp
    debug("HNSW index created%s\n", "");
    
    
    // size_t nq; // num queries
    // std::vector<float> queries;

    { // populating the database
        printf("====================Vectors====================\n");
       
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float* xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0, N, d, nb);

        // index->add(nb, xb);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        base_index.add(N, xb);
        printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);


        delete[] xb;       
    }

    size_t nq;
    float* xq;
    { // queries
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as expected 128");
    }

    // see demo_sift1M to get ground truth and record recall
    

    { // print out stats
        printf("====================================\n");
        printf("============ BASE INDEX =============\n");
        printf("====================================\n");
        base_index.printStats();
        printf("====================================\n");
        printf("============ HYBRID INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats();
    }

    { // get index size
        printf("====================================\n");
        printf("============ INDEX SIZES=============\n");
        printf("====================================\n");
        /* for base index */ 
        //  file name
        std::ostringstream ss;
        ss << "./tmp/base_hnsw_N=" << N << ".faissindex";
        std::string s_tmp = ss.str();
        const char* outfilename = s_tmp.c_str();
        printf("[%.3f s] storing the hnsw index to %s\n",
               elapsed() - t0,
               outfilename);
        // write index to disk
        write_index(&base_index, outfilename);
        //  measure file size
        std::ifstream in_file(outfilename, std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int file_size = in_file.tellg();
        std::cout<<"====Size of the base index is"<<" "<< file_size<<" "<<"bytes" << std::endl;

         /* for hybrid index */ 
        //  file name
        std::ostringstream ss2;
        ss2 << "./tmp/hybrid=" << gamma << "_hnsw_N=" << N << ".faissindex";
        s_tmp = ss2.str();
        outfilename = s_tmp.c_str();
        printf("[%.3f s] storing the hnsw index to %s\n",
               elapsed() - t0,
               outfilename);
        // write index to disk
        write_index(&hybrid_index, outfilename);
        //  measure file size
        std::ifstream in_file2(outfilename, std::ios::binary);
        in_file2.seekg(0, std::ios::end);
        file_size = in_file2.tellg();
        std::cout<<"====Size of the hybrid index is"<<" "<< file_size<<" "<<"bytes" << std::endl;
        
    }
    
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();
    
    { // searching the base database
        printf("====================BASE INDEX====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        int k_prime = k * 1;
        std::vector<faiss::idx_t> nns(k_prime * nq);
        std::vector<float> dis(k_prime * nq);

        double t1 = elapsed();
        base_index.search(nq, xq, k_prime, dis.data(), nns.data());
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < 99; i++) {
            printf("query %2d nn's: ", i);
            for (int j = 0; j < k_prime; j++) {
                printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k_prime; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2 - t1);


    }

    { // searching the hybrid database
        printf("==================== HYBRID INDEX ====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        double t1 = elapsed();
        int filter = 1;
        hybrid_index.search(nq, xq, k, dis.data(), nns.data(), filter);
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < 99; i++) {
            printf("query %2d nn's: ", i);
            for (int j = 0; j < k; j++) {
                printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2 - t1);

    }

    {
        //measure query times again x times
        int x = 10;
        std::vector<float> base_queries(x);
        std::vector<float> hybrid_queries(x);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        for (int i = 0; i < x; i++) {
            double t1 = elapsed();
            base_index.search(nq, xq, k, dis.data(), nns.data());
            double t2 = elapsed();
            base_queries[i] = t2 - t1;

            double t3 = elapsed();
            int filter = 1;
            hybrid_index.search(nq, xq, k, dis.data(), nns.data(), filter);
            double t4 = elapsed();
            hybrid_queries[i] = t4 - t3;

        }

        std::cout << "Base Query Times: ";
        for (int i = 0; i < x; i++) {
            std::cout << base_queries[i] << ", ";
        } 
        std::cout << std::endl;
        
        std::cout << "Hybrid Query Times: ";
        for (int i = 0; i < x; i++) {
            std::cout << hybrid_queries[i] << ", ";
        } 
        std::cout << std::endl;

        // std::cout << "Hybrid Query Times: " << hybrid_queries << "\n" << std::endl;


        
    }

    printf("-----DONE-----\n");
}