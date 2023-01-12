#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

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
    printf("====================\nSTART: running demo_test_HYBRID_LARGE for hnsw...\n");
    double t0 = elapsed();
    // int opt;
    
    int efc = 40; // default is 40
    int efs = 16; //  default is 16
    int k = 10; // search parameter
    int d = 128; // dimension of the vectors to index
    int M = 32; // HSNW param M
    size_t nb = 1000; // size of the database we plan to index
    float attr_sel = 0.1;
    int gamma = (int) 1 / attr_sel;


    // generate metadata
    std::vector<int> metadata(nb);
    for (int i = 0; i < nb; i++) {
        int rand_num = rand() % gamma;
        metadata[i] = rand_num;
    }
    assert(nb == metadata.size());

    // print metadata
    printf("[%.3f s] Set metadata, %ld attr's generated\n", 
        elapsed() - t0, metadata.size());
    printf("Metadata: ");
    for (int i = 0; i < nb; i++) {
        printf("%d, ", metadata[i]);
    }
    printf("\n");
    

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %d, M: %d, nb: %ld, gamma: %d\n",
               elapsed() - t0, d, M, nb, gamma);
    // base index
    faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
    base_index.hnsw.efConstruction = efc; // default is 40 * gamma in HNSW.capp
    base_index.hnsw.efSearch = efs; // default is 16 * gamma in HNSW.capp
    // hybrid index
    faiss::IndexHNSWHybrid hybrid_index(d, M, gamma, metadata);
    hybrid_index.hnsw.efConstruction = efc; // default is 40 * gamma in HNSW.capp
    hybrid_index.hnsw.efSearch = efs; // default is 16 * gamma in HNSW.capp
    debug("HNSW index created%s\n", "");
    
    // initialize some things
    std::mt19937 rng(0); // random generator to be used for creating vectors, set seed 0

    size_t nq; // num queries
    std::vector<float> queries;

    { // populating the database
        printf("====================Vectors====================\n");
        printf("[%.3f s] Building a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);

        std::vector<float> database(nb * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        base_index.add(nb, database.data());
        printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);

        hybrid_index.add(nb, database.data());
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);

        // TODO: print out stats here
        // printf("[%.3f s] imbalance factor: %g\n",
        //        elapsed() - t0,
        //        index.invlists->imbalance_factor());

        // remember a few elements from the database as queries
        int i0 = 3;
        int i1 = 5;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }


    
    

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
        ss << "./tmp/index_hnsw_N=" << nb << ".faissindex";
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
        ss2 << "./tmp/index_hnsw_N=" << nb << ".faissindex";
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

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        double t1 = elapsed();
        base_index.search(nq, queries.data(), k, dis.data(), nns.data());
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < nq; i++) {
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
        hybrid_index.search(nq, queries.data(), k, dis.data(), nns.data(), filter);
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < nq; i++) {
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

    printf("-----DONE-----\n");
}