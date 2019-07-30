
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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include "thrust_rmm_allocator.h"

//#define DUMP_TO_FILE

#define MB (1024*1024)

#include "cTimer.h"
cTimer timer;

#include "cClipping.h"

typedef rmm::device_vector<float>::iterator IterFloat;
typedef rmm::device_vector<int>::iterator IterInt;

/**
    Clips the input vector given the input-defined plane and copy the output to posOut

	@param 
		posIn: input vector to be clipped
		normal: normal of the plane with which posIn will be clipped
		d: the point along with the nomral with which posIn will be clipped
		posOut: output vector from the clipping
		size: size of the posIn vector
		threadId: thread id with which clipping will be ran
*/

void  clip (rmm::device_vector<float> *posIn, float *normal, float d,rmm::device_vector<float> *posOut, size_t size, int threadId)
{
	plane_clippingPDB clipPDB	(normal, d);

	rmm::device_vector<int> clipFlag	( size, -1  ); // num vertices

	// Define the iterators on posIn
	strided_range<IterFloat> X			( posIn->begin()  , posIn->end(), 4);
	strided_range<IterFloat> Y			( posIn->begin()+1, posIn->end(), 4);
	strided_range<IterFloat> Z			( posIn->begin()+2, posIn->end(), 4);
	strided_range<IterFloat> W			( posIn->begin()+3, posIn->end(), 4);

	strided_range<IterInt> clipX	( clipFlag.begin(),	clipFlag.end(), 4);
	strided_range<IterInt> clipY	( clipFlag.begin()+1, clipFlag.end(), 4);
	strided_range<IterInt> clipZ	( clipFlag.begin()+2, clipFlag.end(), 4);
	strided_range<IterInt> clipW	( clipFlag.begin()+3, clipFlag.end(), 4);

	// Apply clipPDB of each element of tuples defined using the iterators
	thrust::for_each (thrust::make_zip_iterator ( thrust::make_tuple( X.begin(), Y.begin(), Z.begin(), W.begin(),
																	   clipX.begin(), clipY.begin(), clipZ.begin (), clipW.begin() ) ),
					   thrust::make_zip_iterator ( thrust::make_tuple( X.end(), Y.end(), Z.end(), W.end(),
																	   clipX.end(), clipY.end(), clipZ.end (), clipW.end() ) ),
					   clipPDB
					  );

	size_t numNotClipped = thrust::count_if(clipX.begin(), clipX.end(), not_clipped<float>());

	// Resize the posOut vector to the correct size and copy the information to posOut
	posOut->resize(numNotClipped * 4);
	thrust::copy_if( posIn->begin(), posIn->end(), clipFlag.begin(), posOut->begin(), not_clipped<float>());
}


int main (int argc, char *argv[])
{
	unsigned int i, iter = 30;
	size_t sx = 400, sy = 400, sz = 2000;
	size_t numParticles = 0;
	std::vector<float> pos; // particle positions

	rmm::device_vector<float> d_pos; // particle positions in GPU
	rmm::device_vector<float> d_posOut; // particle positions out in GPU

	double elapsed = 0.0;
	double totalElapsed = 0.0;
	double transferElapsed = 0.0;
	double computeElapsed = 0.0;

	// This willl be used to generate plane's normals randomly
	// between -1 to 1
	std::mt19937 rng(time(NULL));
	std::uniform_real_distribution<float> gen(-1.0, 1.0);


	if (argc < 5)
	{
		std::cout << "Usage: clipping x_size y_size z_size  iterations \n";
		return 1;
	}
	sx = std::stoll (std::string(argv[1]));
	sy = std::stoll (std::string(argv[2]));
	sz = std::stoll (std::string(argv[3]));
	iter = std::stoi (std::string(argv[4]));
	numParticles = sx*sy*sz;

	std::cout << "Domain size is " << sx << " x " << sy << " x " << sz << " = " << numParticles << " particles" << std::endl;
	std::cout << "Size MB: " << (sizeof(float) * numParticles * 4.0) / MB <<std::endl;
	std::cout << "Iterations: " << iter << std::endl;
	std::cout << "Generating particles...\n";

	// Types of allocations:
	// CudaDefaultAllocation
	// PoolAllocation
	// CudaManagedMemory

	rmmOptions_t options{rmmAllocationMode_t::PoolAllocation, 0, true};
	rmmInitialize(&options);

	// Timer to record time taken to initialize dataset
	timer.reset ();

	// Synthesize random data of given size
	initDataset(&pos, sx, sy, sz);
	std::cout << timer.getElapsedMilliseconds() << " ms\n";
	std::cout << "done!\n";
	timer.reset ();


	// plane defined by normal and D
	float normal[3], d = 0.0f;
	std::cout << "Clipping domain...\n";

	for (i=0;i<iter;i++)
	{
		// Generating plane's normals randomly
		// between -1 to 1
		normal[0] = gen(rng);
		normal[1] = gen(rng);
		normal[2] = gen(rng);

		// Timer to record H to D transfer time
		timer.reset();
		d_pos = pos;
		elapsed = timer.getElapsedMilliseconds();
		std::cout << "H to D: " << elapsed << " ms\n";
		//std::cout << d_pos.size();
		transferElapsed += elapsed;

		// Compute clipping
		timer.reset ();
		clip(&d_pos, normal, d, &d_posOut, d_pos.size(), 0);
		elapsed = timer.getElapsedMilliseconds();
		std::cout << "Particles_out " << d_posOut.size() << " clipping in " << elapsed << "ms\n";
		computeElapsed += elapsed;

		timer.reset();

		std::vector<float> posOut(d_posOut.size());
		thrust::copy(d_posOut.begin(), d_posOut.end(), posOut.begin());
		
		// Clear d_pos, d_posOut vectors
		d_pos.clear();
		rmm::device_vector<float>().swap(d_pos);
		d_posOut.clear();
		rmm::device_vector<float>().swap(d_posOut);
		
		// Timer to record D to H transfer time
		elapsed = timer.getElapsedMilliseconds();
		std::cout << "D to H: " << elapsed << " ms\n";
		transferElapsed += elapsed;

#ifdef DUMP_TO_FILE
		dump (posOut, posOutSize*4);
#endif


	}
	totalElapsed = computeElapsed + transferElapsed;
	std::cout << "Total transfers Avg time (ms) after " << iter << " iterations " << transferElapsed / iter << std::endl;
	std::cout << "Total compute Avg time (ms) after " << iter << " iterations " << computeElapsed / iter << std::endl;
	std::cout << "Total Avg time (ms) after " << iter << " iterations " << totalElapsed / iter << std::endl;

	return 0;
}
