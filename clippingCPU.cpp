
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

//#define DUMP_TO_FILE

#define MB (1024*1024)

#include "cTimer.h"
cTimer timer;

#include "cClipping.h"

typedef std::vector<float>::iterator IterFloat;
typedef std::vector<int>::iterator IterInt;

void  clip (std::vector<float> *posIn, float *normal, float d, float* &posOut, size_t size, size_t *sizeOut, int threadId)
{
	plane_clippingPDB clipPDB	(normal, d);

	std::vector<int> clipFlag	( size, -1  ); // num vertices


	strided_range<IterFloat> X			( posIn->begin()  , posIn->end(), 4);
	strided_range<IterFloat> Y			( posIn->begin()+1, posIn->end(), 4);
	strided_range<IterFloat> Z			( posIn->begin()+2, posIn->end(), 4);
	strided_range<IterFloat> W			( posIn->begin()+3, posIn->end(), 4);

	strided_range<IterInt> clipX	( clipFlag.begin(),	clipFlag.end(), 4);
	strided_range<IterInt> clipY	( clipFlag.begin()+1, clipFlag.end(), 4);
	strided_range<IterInt> clipZ	( clipFlag.begin()+2, clipFlag.end(), 4);
	strided_range<IterInt> clipW	( clipFlag.begin()+3, clipFlag.end(), 4);

	thrust::for_each ( thrust::make_zip_iterator ( thrust::make_tuple( X.begin(), Y.begin(), Z.begin(), W.begin(),
																	   clipX.begin(), clipY.begin(), clipZ.begin (), clipW.begin() ) ),
					   thrust::make_zip_iterator ( thrust::make_tuple( X.end(), Y.end(), Z.end(), W.end(),
																	   clipX.end(), clipY.end(), clipZ.end (), clipW.end() ) ),
					   clipPDB
					  );

	size_t numNotClipped = thrust::count_if(clipX.begin(), clipX.end(), not_clipped<float>());
	*sizeOut = numNotClipped;
	size_t size4 = numNotClipped * 4;

	//std::cout << "not clipped " << *sizeOut << std::endl;

    posOut = new float[size4];

    thrust::copy_if( posIn->begin(), posIn->end(), clipFlag.begin(), posOut, not_clipped<float>());
}


int main (int argc, char *argv[])
{
	unsigned int i, iter = 30;
	size_t sx = 400, sy = 400, sz = 1000;
	size_t numParticles = 0;
	std::vector<float> pos; // particle positions
	float *posOut = 0;
	size_t posOutSize = 0;

	double totalElapsed = 0.0;
	
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

	timer.reset ();
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
		std::cout << "N=" << normal[0] << "," <<  normal[1] << "," << normal[2] <<  " ";

		timer.reset ();
		clip(&pos, normal, d, posOut, pos.size(), &posOutSize, 0);
		std::cout << "Particles_out " << posOutSize << " " << timer.getElapsedMilliseconds() << " ms\n";
		totalElapsed += timer.getElapsedMilliseconds();

#ifdef DUMP_TO_FILE
		dump (posOut, posOutSize*4);
#endif
		if (posOut)
			delete [] posOut;
		posOutSize = 0;
	}
	std::cout << "Avg time (ms) after " << iter << " iterations " << totalElapsed / iter << std::endl;

	return 0;
}
