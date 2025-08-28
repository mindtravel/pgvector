# g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_flat.cpp -lpq && ./cpp/vector_search -p 1
# g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_jl.cpp -lpq && ./cpp/vector_search -p 1
g++ -std=c++17 -I/usr/include/postgresql -o cpp/vector_search cpp/parallel_pq.cpp -lpq && ./cpp/vector_search -p 1