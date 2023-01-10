#! /bin/bash

# make -C build demo_new_test
# ./build/demos/demo_new_test 1000 32 1

make -C build demo_test_search
# ./build/demos/demo_test_search 100 32 1
./build/demos/demo_test_search 10 3 1 &> log.txt #