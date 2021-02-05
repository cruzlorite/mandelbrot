/*
 * Copyright (C) 2020 José María Cruz Lorite
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <CL/cl.hpp>

#include "util/lodepng.h"

// OpenCL Mandelbrot kernel
const std::string kernel = R"(
    __kernel
    void mandelbrot(
            __global uchar4* img,       // Output image
            __global uchar4* palette,   // Color palette
            int paletteSize,            // Number of palette colors
            float2 min,                 // Complex min value
            float2 delta,               // max - min
            uint maxDepth,              // Max iterations
            float scaleForce)           // Modifies how color palette is useds
    {
        // Get global invocation id
        int x = get_global_id(0);
        int y = get_global_id(1);

        // Get image width and height
        int width  = get_global_size(0);
        int height = get_global_size(1);

        // Interpolate complex value
        float2 c = min + delta * (float2)(x / (float)(width - 1),
                                          y / (float)(height - 1));

        int depth = 0;
        float2 z = c;
        float2 z2 = z * z;
        for(; depth < maxDepth && z2.x + z2.y < 4.0f; ++depth) {
            z = (float2)(z2.x - z2.y, 2 * z.x * z.y) + c;
            z2 = z * z;
        }        

        if (depth == maxDepth) {
            img[y * width + x] = (uchar4)(0, 0, 0, 255);
        } else {
            // Iteration normalized
            float i = (float)(depth) / (float)(maxDepth - 1);
            float ie = fmin(1.0f, log(i * scaleForce + 1.0f) / log(scaleForce));

            int index = floor((1.0f - ie) * (float)(paletteSize - 1));
            img[y * width + x] = palette[index];
        }
    };
)";

// Check OpenCL error
void checkCLError(cl_int status, const std::string msg)
{
    if (status != CL_SUCCESS) {
        std::cerr << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Check program building
void checkCLProgramBuildError(
    cl_int status,
    const cl::Program& program,
    const std::vector<cl::Device>& devices) {

    if (status != CL_SUCCESS) {
        std::cout << "Error building program." << std::endl;

        for (const cl::Device& dev : devices) {
            // Get the build log
            std::string name     = dev.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            std::cerr << "Build log: " << name << ":" << std::endl << buildlog << std::endl;
        }

        exit(EXIT_FAILURE);
    } 
}

// Print device info
void printDeviceInfo(const cl::Device& dev) {
    std::cout << dev.getInfo<CL_DEVICE_VENDOR>() << ", ";
    std::cout << dev.getInfo<CL_DEVICE_NAME>() << ", ";
    std::cout << dev.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    std::cout << "Compute units:        " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Max work-items:       " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    std::cout << "Max work-group sizes: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << " x " <<
                                             dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << " x " << 
                                             dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[2] << std::endl;
}

int main() {
    /******** PARAMETERS *********/
    cl_uint  localGroupSize = 32;
    cl_uint  maxDepth       = 1024;
    cl_uint  width          = 1024 * 1;
    cl_uint  height         = 1024 * 1;
    cl_float scaleForce     = 20.0f;
    /*****************************/

    // Allocate space for the image on host
    std::vector<cl_uchar> img(width * height * 4);

    // Load color palette
    unsigned int w_palette = 0; // 1
    unsigned int h_palette = 0; // 512
    std::vector<cl_uchar> palette;
    lodepng::decode(palette, w_palette, h_palette, "palette.png");

    // Complex plane min max
    float min_real = -2.0;
    float max_real = 1.0f;
    float min_imag = ((max_real - min_real) * height / width) / -2.0f;
    float max_imag = -1.0f * min_imag;

    cl_float2 min {min_real, min_imag};
    cl_float2 delta {max_real - min_real, max_imag - min_imag};

    cl_int status;

    // Get start time
    auto start = std::chrono::high_resolution_clock::now();

    // Query for platforms
    std::vector<cl::Platform> platforms;
    status = cl::Platform::get(&platforms);
    checkCLError(status, "Error getting platform.");

    // Get list of GPU devices
    std::vector<cl::Device> devices;
    status = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    checkCLError(status, "Error getting devices.");

    // Print device info
    for (cl::Device dev : devices) {
        printDeviceInfo(dev);
        std::cout << std::endl;
    }

    // Create context
    cl::Context context(devices);

    // Create command-queue
    cl::CommandQueue queue(context, devices[0]);

    // Create buffers
    cl::Buffer imgBuffer(context, CL_MEM_WRITE_ONLY, img.size());
    cl::Buffer paletteBuffer(context, CL_MEM_READ_ONLY, palette.size());

    // Write palette buffer
    status = queue.enqueueWriteBuffer(paletteBuffer, CL_TRUE, 0, palette.size(), palette.data());
    checkCLError(status, "Error writing color palette.");

    // Creating and compiling the OpenCl kernel
    cl::Program::Sources sources(1, std::make_pair(kernel.c_str(), kernel.size()));
    cl::Program program(context, sources);

    // Compile for available devices
    status = program.build(devices);
    checkCLProgramBuildError(status, program, devices);

    // Extract the kernel from the program
    cl::Kernel kernel(program, "mandelbrot");

    // Set kernel arguments
    status  = kernel.setArg(0, imgBuffer);
    status |= kernel.setArg(1, paletteBuffer);
    status |= kernel.setArg(2, h_palette);
    status |= kernel.setArg(3, min);
    status |= kernel.setArg(4, delta);
    status |= kernel.setArg(5, maxDepth);
    status |= kernel.setArg(6, scaleForce);
    checkCLError(status, "Error setting kernel arguments.");

    // Define index space of work items
    cl::NDRange global(width, height);
    cl::NDRange local(localGroupSize, localGroupSize);

    // Dispach kernel
    status = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    checkCLError(status, "Error enqueuing kernel.");

    // Copy output data back
    status = queue.enqueueReadBuffer(imgBuffer, CL_TRUE, 0, img.size(), img.data());
    checkCLError(status, "Error rading buffer.");

    // Get end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f <<
        " seconds" << std::endl;

    // To file
    lodepng::encode("mandelbrot.png", img.data(), width, height);

    exit(EXIT_SUCCESS);
}
