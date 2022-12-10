#! /bin/bash

cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -B build
make -C build -j faiss
make -C build demo_ivfpq_indexing
./build/demos/demo_ivfpq_indexing