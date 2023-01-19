#! /bin/bash

# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -B build
# make -C build -j faiss
# make -C build demo_new_test
# ./build/demos/demo_new_test 1000 32 1


cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -B build
make -C build -j faiss

make -C build demo_test_search
make -C build demo_test_search_small
make -C build demo_test_hybrid_small
make -C build demo_test_hybrid_large
make -C build make_indices


# executable to run
# ./build/demos/demo_test_search 100 32 1
# ./build/demos/demo_test_search 10 3 1 &> log.txt
# ./build/demos/demo_test_search_small
# ./build/demos/demo_test_hybrid_large &> 1000_test.txt

# ./build/demos/make_indices 1000 &> indices.txt
# ./build/demos/make_indices 10000 &>> indices.txt
# ./build/demos/make_indices 100000 &>> indices.txt
# ./build/demos/make_indices 1000000 &>> indices.txt
# ./build/demos/make_indices 10000000 &> 1mil_test.txt
# ./build/demos/make_indices 100000 &> 1k_gamma_test.txt
# ./build/demos/make_indices 1000000 &> 1mil_gamma1k_test.txt

# n=10k tests
# ./build/demos/make_indices 10000 10
./build/demos/make_indices 10000 10 &> n=10k_gamma=10_test.txt 
./build/demos/make_indices 10000 40 &> n=10k_gamma=40_test.txt 
./build/demos/make_indices 10000 70 &> n=10k_gamma=70_test.txt 
./build/demos/make_indices 10000 100 &> n=10k_gamma=1h_test.txt 

# n=100k tests
./build/demos/make_indices 100000 10 &> n=100k_gamma=10_test.txt 
./build/demos/make_indices 100000 40 &> n=100k_gamma=40_test.txt 
./build/demos/make_indices 100000 70 &> n=100k_gamma=70_test.txt 
./build/demos/make_indices 100000 100 &> n=100k_gamma=1h_test.txt 
