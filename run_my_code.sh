#! /bin/bash

cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -B build
make -C build -j faiss
make -C build demo_new_test
./build/demos/demo_new_test 1000 32