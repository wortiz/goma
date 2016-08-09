#include <fstream>
#include <iostream>

#include <stdlib.h>
#include <cuda_runtime.h>
extern "C" {
void *cudaCallocWrap(size_t nmemb, size_t size)
{
  void *mem;
  size_t total_size = nmemb * size;
  cudaMallocManaged(&mem, total_size);
  memset(mem, 0, total_size);
  return mem;
}

void *cudaMallocWrap(size_t size)
{
  void *mem;
  cudaMallocManaged(&mem, size);
  return mem;
}


void copy_file_cpp(const char *infile, const char *outfile) {
  std::ifstream  src(infile, std::ios::binary);
  std::ofstream  dst(outfile,   std::ios::binary);
  dst << src.rdbuf();
}

}
