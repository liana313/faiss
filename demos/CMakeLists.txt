# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)

add_executable(demo_new_test EXCLUDE_FROM_ALL demo_new_test.cpp)
target_link_libraries(demo_new_test PRIVATE faiss)

add_executable(demo_test_search EXCLUDE_FROM_ALL demo_test_search.cpp)
target_link_libraries(demo_test_search PRIVATE faiss)

add_executable(demo_test_search_small EXCLUDE_FROM_ALL demo_test_search_small.cpp)
target_link_libraries(demo_test_search_small PRIVATE faiss)

add_executable(demo_test_hybrid_small EXCLUDE_FROM_ALL demo_test_hybrid_small.cpp)
target_link_libraries(demo_test_hybrid_small PRIVATE faiss)

add_executable(demo_test_hybrid_large EXCLUDE_FROM_ALL demo_test_hybrid_large.cpp)
target_link_libraries(demo_test_hybrid_large PRIVATE faiss)

add_executable(make_indices EXCLUDE_FROM_ALL make_indices.cpp)
target_link_libraries(make_indices PRIVATE faiss)

add_executable(run_queries EXCLUDE_FROM_ALL run_queries.cpp)
target_link_libraries(run_queries PRIVATE faiss)
