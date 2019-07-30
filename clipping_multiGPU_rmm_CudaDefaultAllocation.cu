
#include <algorithm>
#include <iostream>
#include <numeric>
#include <array>
#include <vector>
#include <stdlib.h>
#include <random>
#include <thread>

#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>

#include "cClipping.h"
#include "cTimer.h"
#include "thrust_rmm_allocator.h"

cTimer timer;

typedef rmm::device_vector<float>::iterator IterFloat;
typedef rmm::device_vector<int>::iterator IterInt;

#define MB (1024*1024)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
    Clips the input vector given the input-defined plane 

	@param 
		posIn: input vector to be clipped
		normal: normal of the plane with which posIn will be clipped
		d: the point along with the nomral with which posIn will be clipped
*/

void clip (rmm::device_vector<float> *posIn, float *normal, float d)
{
	plane_clippingPDBver2 clip	(normal, d);

	strided_range<IterFloat> X		( posIn->begin()  , posIn->end(), 4);
	strided_range<IterFloat> Y		( posIn->begin()+1, posIn->end(), 4);
	strided_range<IterFloat> Z		( posIn->begin()+2, posIn->end(), 4);
	strided_range<IterFloat> W		( posIn->begin()+3, posIn->end(), 4);

	size_t new_size = thrust::remove_if( thrust::make_zip_iterator ( thrust::make_tuple( X.begin(), Y.begin(), Z.begin(), W.begin() )),
			   	   	   	   	   	   	   	 thrust::make_zip_iterator ( thrust::make_tuple( X.end(),Y.end(), Z.end(), W.end() )),
			   	   	   	   	   	   	   	 clip )
	                      	  	  	  	- thrust::make_zip_iterator(thrust::make_tuple(X.begin(), Y.begin(), Z.begin(), W.begin() ));

	// Resize the input vector to appropriate size
	posIn->resize(new_size*4);
}

/**
    Copy vec to d_vec on the given thread

	@param 
		devId: stream ID used to copy the vectors
		vec: vector to be copied from
		d_vec: vector to be copied to
*/

void memCopyHtoD (int devId, std::vector<float> *vec, rmm::device_vector<float> *d_vec )
{
	gpuErrchk(cudaSetDevice(devId));
	*d_vec = *vec;
}

/**
    Copy vec to d_vec on the given thread

	@param 
		devId: stream ID used to copy the vectors
		vec: vector to be copied from
		d_vec: vector to be copied to
*/

void memCopyDtoH (int devId, std::vector<float> *vec, rmm::device_vector<float> *d_vec )
{
	gpuErrchk(cudaSetDevice(devId));
	vec->resize(d_vec->size());
	thrust::copy(d_vec->begin(), d_vec->end(), vec->begin());
}

/**
    Run clipping on given thread

	@param 
		devId: stream ID used to run the clipping
		d_pos: vector to be clipped
		normal: normal of the plane with which d_pos will be clipped
		d: the point along with the nomral with which d_pos will be clipped
*/
void launch (int devId, rmm::device_vector<float> *d_pos, float *normal, float d)
{
	gpuErrchk(cudaSetDevice(devId));
	clip (d_pos, normal, d);

}

int main(int argc, char *argv[])
{
    size_t sx, sy, sz;
    int numStreams, i, j;

    int deviceCount = 0;
    unsigned int iter = 0;

	double elapsed = 0.0;
	double totalElapsed = 0.0;
	double htodElapsed = 0.0;
	double dtohElapsed = 0.0;
	double transferElapsed = 0.0;
	double computeElapsed = 0.0;

    // This willl be used to generate plane's normals randomly
    // between -1 to 1
    std::mt19937 rng(time(NULL));
	std::uniform_real_distribution<float> gen(-1.0, 1.0);
	
    // plane defined by normal and D
    float normal[3], d = 0.5f;

    if (argc < 5){
        std::cout << "Usage: clipping x_size y_size z_size iterations" << std::endl;
        return 1;
    }
    sx = std::stoll (std::string(argv[1]));
    sy = std::stoll (std::string(argv[2]));
    sz = std::stoll (std::string(argv[3]));
    iter = std::stoi (std::string(argv[4]));

	size_t numParticles = sx*sy*sz;
	
	// Retrieve the number of streams/devices we can use
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    numStreams = deviceCount;
    size_t numParticlesPerThread = sx*sy*sz/numStreams;

    std::cout << "========\n";
	std::cout << "Domain size is " << sx << " x " << sy << " x " << sz << " = " << numParticles << " particles" << std::endl;
	std::cout << "Size MB " << (sizeof(float) * numParticles * 4.0) / MB <<std::endl;
	std::cout << "Num. Devices " << deviceCount << std::endl;
	std::cout << "Particles per device " << numParticlesPerThread << std::endl;
	std::cout << "Size MB per device " << (sizeof(float) * numParticlesPerThread * 4.0) / MB <<std::endl;

	// Define array of vectors and threads to be used 
    std::thread thread[numStreams];
	std::vector <float> pos[numStreams];
	std::vector <float> posOut[numStreams];
	rmm::device_vector<float> d_pos[numStreams];

	std::cout << "Generated particles...\n";
	
	// Types of allocations:
	// CudaDefaultAllocation
	// PoolAllocation
	// CudaManagedMemory

	rmmOptions_t options{rmmAllocationMode_t::CudaDefaultAllocation, 0, true};
	rmmInitialize(&options);
	
	// Timer to record time taken to initialize dataset
	timer.reset();

	// Launch threads and join
	for (i=0;i<numStreams;i++)
	{
		size_t szMin =  i*(sz/numStreams);
		size_t szMax = (i+1)*(sz/numStreams);
		thread[i] =std::thread (initDatasetChunk, &pos[i],  sx, sy, szMin, szMax);
	}
	for(i = 0; i < numStreams; i++)
	{
		thread[i].join ();
	}
	std::cout << "in " << timer.getElapsedMilliseconds() << " ms\n";


	for(j=0;j<iter;j++)
	{
		// Generating plane's normals randomly
		// between -1 to 1
		normal[0] = gen(rng);
		normal[1] = gen(rng);
		normal[2] = gen(rng);

		// Copy H to D
		timer.reset ();
		for (i=0;i<numStreams;i++)
		{
			thread[i] =std::thread (memCopyHtoD, i, &pos[i], &d_pos[i]);
		}
		for(i = 0; i < numStreams; i++)
		{
			thread[i].join ();
		}
		elapsed = timer.getElapsedMilliseconds();
		std::cout << "H to D: " << elapsed << " ms\n";
		htodElapsed+=elapsed;
		transferElapsed+=elapsed;

		// launch the clip kernel
		timer.reset ();
		for(i = 0; i < numStreams; i++)
		{
			thread[i] =std::thread (launch, i, &d_pos[i], normal, d );
		}
		for(i = 0; i < numStreams; i++)
		{
			thread[i].join ();
		}
		elapsed = timer.getElapsedMilliseconds();
		//std::cout << "Clipping: " << elapsed << " ms\n";
		computeElapsed += elapsed;

		// Copy D to H
		timer.reset ();
		for (i=0;i<numStreams;i++)
		{
			thread[i] =std::thread (memCopyDtoH, i, &posOut[i], &d_pos[i]);
		}
		for(i = 0; i < numStreams; i++)
		{
			thread[i].join ();
		}
		elapsed = timer.getElapsedMilliseconds();
		//std::cout << "D to H: " << elapsed << " ms\n";
		dtohElapsed+=elapsed;
		transferElapsed+=elapsed;
	}
	std::cout << "--------\n";
	totalElapsed = computeElapsed + transferElapsed;
	std::cout << "H to D Avg time (ms) after " << iter << " iterations " << htodElapsed / iter << std::endl;
	std::cout << "D to H Avg time (ms) after " << iter << " iterations " << dtohElapsed / iter << std::endl;
	std::cout << "Transfers Avg time (ms) after " << iter << " iterations " << transferElapsed / iter << std::endl;
	std::cout << "Compute Avg time (ms) after " << iter << " iterations " << computeElapsed / iter << std::endl;
	std::cout << "Total Avg time (ms) after " << iter << " iterations " << totalElapsed / iter << std::endl;

    return 0;

}



