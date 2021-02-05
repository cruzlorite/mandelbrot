// Copyright(C) 2020 José María Cruz Lorite
//
// This file is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this project.  If not, see <https://www.gnu.org/licenses/>.

#include <chrono>
#include <iostream>
#include <vector>

// CUDA helper_math.h from NVIDIA repository on github
// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h
#include "helper_math.h"

// export/import png images
#include "util/lodepng.h"

// Handy macro
#define check(func, msg) __check((func), msg, __FILE__, __LINE__)

// Check if something goes wrong
void __check(cudaError_t err, const std::string& msg, const std::string& file, int line) {
    if (err != cudaSuccess) {
        std::cerr << msg << ". File '" << file << "', line " << line <<std::endl;
        exit(EXIT_FAILURE);
    }
}

// Print properties for particular device
void printDeviceProp(int dev) {
    // get device properties
    cudaDeviceProp prop;
    check( cudaGetDeviceProperties(&prop, dev), "Error getting device properties");

    std::cout << "NVIDIA Corporation, " << prop.name << ", CUDA " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Number of muliprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp size in threads:     " << prop.warpSize<< std::endl;
    std::cout << "Max threads per block:    " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimension:      " << prop.maxThreadsDim[0] << " x " <<
                                                 prop.maxThreadsDim[1] << " x " <<
                                                 prop.maxThreadsDim[2] << ""  << std::endl;
    std::cout << "Max grid dimension:       " << prop.maxGridSize[0] << " x " <<
                                                 prop.maxGridSize[1] << " x " <<
                                                 prop.maxGridSize[2] << ""  << std::endl;
}

// CUDA Mandelbrot kernel
__global__
void mandelbrot(
    uint32_t* img,      // Output image
    int width,          // Image width
    int height,         // Image height 
    uint32_t* palette,  // Color palette
    int paletteSize,    // Number of palette colors
    float2 min,         // Complex min value
    float2 delta,       // max - min
    uint maxDepth,      // Max iterations
    float scaleForce)   // Modifies how color palette is used
{
    // Get global invocation id
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Interpolate complex value for this pixel
    float2 c = min + delta * make_float2(
                                x / (float)(width - 1),
                                y / (float)(height - 1));

    int depth = 0;
    float2 z = c;
    float2 z2 = z * z;
    for(; depth < maxDepth && z2.x + z2.y < 4.0f; ++depth) {
        z = make_float2(z2.x - z2.y, 2 * z.x * z.y) + c;
        z2 = z * z;
    }

    int index = y * width + x;
    if (depth == maxDepth) {
        img[index] = 0xFF000000; // Black
    } else {
        // Iteration normalized
        float i = (float)(depth) / (float)(maxDepth - 1);
        float ie = fmin(1.0f, log(i * scaleForce + 1.0f) / log(scaleForce));

        int paletteIndex = floor((1.0f - ie) * (float)(paletteSize - 1));
        img[index] = palette[paletteIndex];
    }
}

int main(void) {
    /******** PARAMETERS *********/
    int  blockSize      = 32;
    int  maxDepth       = 1024;
    int  width          = 1024 * 1;
    int  height         = 1024 * 1;
    float scaleForce    = 20.0f;
    /*****************************/

    // Allocate space for the image on host
    std::vector<uint8_t> img(width * height * 4);

    // Load color palette from file
    unsigned int width_palette = 0;
    unsigned int paletteSize = 0;
    std::vector<uint8_t> palette;
    lodepng::decode(palette, width_palette, paletteSize, "palette.png");

    // Complex plane min max
    float min_real = -2.0;
    float max_real = 1.0f;
    float min_imag = ((max_real - min_real) * height / width) / -2.0f;
    float max_imag = -1.0f * min_imag;

    float2 min      {min_real, min_imag};
    float2 delta    {max_real - min_real, max_imag - min_imag};

    // On device buffers
    uint32_t *img_dev, *palette_dev;

    // Get start time
    auto start = std::chrono::high_resolution_clock::now();

    // Print CUDA properties
    int count; 
    check( cudaGetDeviceCount(&count), "Error getting device count");

    for (int i = 0; i < count; ++i) {
        printDeviceProp(i);
        std::cout << std::endl;
    }

    // Allocate memopry on device
    check( cudaMalloc(&img_dev, img.size()), "Error allocating memory on device" ); 
    check( cudaMalloc(&palette_dev, palette.size()), "Error allocating memory on device" ); 

    // Copy palette to device
    check( cudaMemcpy(palette_dev, palette.data(), palette.size(), cudaMemcpyHostToDevice),
        "Error copying palette from host to device");

    // Kernel invocation
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

    mandelbrot<<<dimGrid, dimBlock>>>(
        img_dev,
        width,
        height,
        palette_dev,
        paletteSize,
        min,
        delta,
        maxDepth,
        scaleForce);

    // Read back from device
    check( cudaMemcpy(img.data(), img_dev, img.size(), cudaMemcpyDeviceToHost),
        "Error copying image from device to host");

    // Free resources
    cudaFree(img_dev);
    cudaFree(palette_dev);

    // Get end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f <<
        " seconds" << std::endl;

    // To file
    lodepng::encode("mandelbrot.png", img.data(), width, height);

    exit(EXIT_SUCCESS);
}
