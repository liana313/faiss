/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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


int debugFlag = 1;  // set to 1 for moderate detail, set to 2 for more detail

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
#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stderr, fmt, __VA_ARGS__); \
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

int main() {
    printf("====================\nSTART: running tests for hnsw...\n");
    double t0 = elapsed();
    
    int d = 128; // dimension of the vectors to index
    int M = 32; // HSNW param M
    size_t nb = 1000 * 1000; // size of the database we plan to index
    debug("Index Params -- d: %d, M: %d, nb: %ld\n", d, M, nb);
    
    faiss::IndexHNSWFlat index(d, M);
    debug("HNSW index created%s\n", "");
    
    std::mt19937 rng; // random generator to be used for creating vectors

    size_t nq; // num queries
    std::vector<float> queries;

    { // populating the database
        printf("[%.3f s] Building a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);

        std::vector<float> database(nb * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        index.add(nb, database.data());

        printf("[%.3f s] Vectors added\n", elapsed() - t0);

        // TODO: print out stats here
        // printf("[%.3f s] imbalance factor: %g\n",
        //        elapsed() - t0,
        //        index.invlists->imbalance_factor());

        // remember a few elements from the database as queries
        int i0 = 4;
        int i1 = 8;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }

    {
        // print out stats
        index.printStats();
    }

    printf("-----DONE-----\n");
}

// TODO Add this to the end
 // { // I/O demo
    //     const char* outfilename = "/tmp/index_trained.faissindex";
    //     printf("[%.3f s] storing the pre-trained index to %s\n",
    //            elapsed() - t0,
    //            outfilename);

    //     write_index(&index, outfilename);
    // }

//OLD CODE

// #include <faiss/IndexFlat.h>
// #include <faiss/IndexIVFPQ.h>
// #include <faiss/index_io.h>

// double elapsed() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }

// int main() {
//     double t0 = elapsed();

//     // dimension of the vectors to index
//     int d = 128;

//     // size of thes database we plan to index
//     size_t nb = 200 * 1000;

//     // make a set of nt training vectors in the unit cube
//     // (could be the database)
//     size_t nt = 100 * 1000;

//     // make the index object and train it
//     faiss::IndexFlatL2 coarse_quantizer(d);

//     // a reasonable number of centroids to index nb vectors
//     int ncentroids = int(4 * sqrt(nb));

//     // the coarse quantizer should not be dealloced before the index
//     // 4 = nb of bytes per code (d must be a multiple of this)
//     // 8 = nb of bits per sub-code (almost always 8)
//     faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, 4, 8);

//     std::mt19937 rng;

//     { // training
//         printf("[%.3f s] Generating %ld vectors in %dD for training\n",
//                elapsed() - t0,
//                nt,
//                d);

//         std::vector<float> trainvecs(nt * d);
//         std::uniform_real_distribution<> distrib;
//         for (size_t i = 0; i < nt * d; i++) {
//             trainvecs[i] = distrib(rng);
//         }

//         printf("[%.3f s] Training the index\n", elapsed() - t0);
//         index.verbose = true;

//         index.train(nt, trainvecs.data());
//     }

//     { // I/O demo
//         const char* outfilename = "/tmp/index_trained.faissindex";
//         printf("[%.3f s] storing the pre-trained index to %s\n",
//                elapsed() - t0,
//                outfilename);

//         write_index(&index, outfilename);
//     }

//     size_t nq;
//     std::vector<float> queries;

//     { // populating the database
//         printf("[%.3f s] Building a dataset of %ld vectors to index\n",
//                elapsed() - t0,
//                nb);

//         std::vector<float> database(nb * d);
//         std::uniform_real_distribution<> distrib;
//         for (size_t i = 0; i < nb * d; i++) {
//             database[i] = distrib(rng);
//         }

//         printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

//         index.add(nb, database.data());

//         printf("[%.3f s] imbalance factor: %g\n",
//                elapsed() - t0,
//                index.invlists->imbalance_factor());

//         // remember a few elements from the database as queries
//         int i0 = 1234;
//         int i1 = 1243;

//         nq = i1 - i0;
//         queries.resize(nq * d);
//         for (int i = i0; i < i1; i++) {
//             for (int j = 0; j < d; j++) {
//                 queries[(i - i0) * d + j] = database[i * d + j];
//             }
//         }
//     }

//     { // searching the database
//         int k = 5;
//         printf("[%.3f s] Searching the %d nearest neighbors "
//                "of %ld vectors in the index\n",
//                elapsed() - t0,
//                k,
//                nq);

//         std::vector<faiss::idx_t> nns(k * nq);
//         std::vector<float> dis(k * nq);

//         index.search(nq, queries.data(), k, dis.data(), nns.data());

//         printf("[%.3f s] Query results (vector ids, then distances):\n",
//                elapsed() - t0);

//         for (int i = 0; i < nq; i++) {
//             printf("query %2d: ", i);
//             for (int j = 0; j < k; j++) {
//                 printf("%7ld ", nns[j + i * k]);
//             }
//             printf("\n     dis: ");
//             for (int j = 0; j < k; j++) {
//                 printf("%7g ", dis[j + i * k]);
//             }
//             printf("\n");
//         }

//         printf("note that the nearest neighbor is not at "
//                "distance 0 due to quantization errors\n");
//     }

//     return 0;
// }
