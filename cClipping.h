/*
 * cClipping.h
 *
 *  Created on: Aug 15, 2017
 *      Author: benjha
 */

#ifndef CCLIPPING_H_
#define CCLIPPING_H_

#include <iostream>
#include <fstream>


#include <vector_functions.hpp> // cuda math
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>

typedef thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float>> pinnedVector;



void dump (float *pos, float size)
{
	size_t i;
	std::ofstream myfile;
	myfile.open ("/media/benjha/Datasets2/particlesOut.txt");

	for (i=0;i<size;i+=4)
	{
		myfile << pos[i] << " " << pos[i+1] << " " << pos[i+2] << " " << pos[i+3] << " "
			   << pos[i] << " " << pos[i+1] << " " << pos[i+2] << " " << pos[i+3] << " "
			   << pos[i] << " " << pos[i+1] << " " << pos[i+2] << " " << pos[i+3] << " "
			   << pos[i] << " " << pos[i+1]
			   << std::endl;
	}
	myfile.close ();
}

void initDatasetChunk (std::vector<float> *pos, size_t x, size_t y, size_t zMin, size_t szMax)
{
	int i,j,k;
	double Pe;
	std::mt19937 rng(time(NULL));
	std::uniform_real_distribution<float> gen(-4.0, 0.0);

	for (i=-(int)x/2;i<((int)x/2);++i)
	{
		for (j=-(int)y/2;j<((int)y/2);++j)
		{
			for (k=zMin;k<szMax;++k)
			{
				//std::cout << i << " " << j << " " << k << " " << Pe << std::endl;

				Pe = gen(rng);
				pos->push_back(i);
				pos->push_back(j);
				pos->push_back(k);
				pos->push_back(Pe);
			}
		}
	}

}


void initDataset (std::vector<float> *pos, size_t x, size_t y, size_t z)
{
	int i,j,k;
	double Pe;
	std::mt19937 rng(time(NULL));
	std::uniform_real_distribution<float> gen(-4.0, 0.0);
#ifdef DUMP_TO_FILE
	std::ofstream myfile;
	myfile.open ("/media/benjha/Datasets2/particles.txt");
#endif
	for (i=-(int)x/2;i<((int)x/2);++i)
	{
		for (j=-(int)y/2;j<((int)y/2);++j)
		{
			for (k=0;k<z;++k)
			{
				Pe = gen(rng);
				pos->push_back(i);
				pos->push_back(j);
				pos->push_back(k);
				pos->push_back(Pe);
#ifdef DUMP_TO_FILE
				myfile << i << " " << j << " " << k << " " << Pe << " "
					   << Pe << " " << Pe << " " << Pe << " " << Pe << " "
					   << Pe << " " << Pe << " " << Pe << " " << Pe << " "
					   << Pe << " " << Pe << " "
					   << std::endl;
#endif
			}
		}
	}
#ifdef DUMP_TO_FILE
	myfile.close();
#endif
}


void initDataset (pinnedVector *pos, size_t x, size_t y, size_t z)
{
	int i,j,k;
	double Pe;
	std::mt19937 rng(time(NULL));
	std::uniform_real_distribution<float> gen(-4.0, 0.0);
#ifdef DUMP_TO_FILE
	std::ofstream myfile;
	myfile.open ("/media/benjha/Datasets2/particles.txt");
#endif
	for (i=-(int)x/2;i<((int)x/2);++i)
	{
		for (j=-(int)y/2;j<((int)y/2);++j)
		{
			for (k=0;k<z;++k)
			{
				Pe = gen(rng);
				pos->push_back(i);
				pos->push_back(j);
				pos->push_back(k);
				pos->push_back(Pe);
#ifdef DUMP_TO_FILE
				myfile << i << " " << j << " " << k << " " << Pe << " "
					   << Pe << " " << Pe << " " << Pe << " " << Pe << " "
					   << Pe << " " << Pe << " " << Pe << " " << Pe << " "
					   << Pe << " " << Pe << " "
					   << std::endl;
#endif
			}
		}
	}
#ifdef DUMP_TO_FILE
	myfile.close();
#endif
}


__host__ __device__ float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z*b.z;
}

__host__ __device__ float length (float3 &v)
{
	return sqrtf(dot(v,v));
}

__host__ __device__ float3 normalize(const float3 &v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

struct plane_clipping
{
	float3		m_normal;
	float		m_d;
	float		m_NormalMag;

	plane_clipping ( float* normal, float D ) // plane defined by Normal and D (distance from origin)
	{
		m_normal 	= make_float3 (normal[0], normal[1], normal[2]);
		m_normal	= normalize (m_normal);
		m_NormalMag	= length (m_normal);
		m_d			= D;
	}

	template <typename Tuple>
	inline __host__ __device__ void operator () (Tuple t)
	{
		float x = thrust::get<0>(t);
		float y = thrust::get<1>(t);
		float z = thrust::get<2>(t);

		int c = clip (make_float3 (x,y,z));

		thrust::get<3>(t) = c;
		thrust::get<4>(t) = c;
		thrust::get<5>(t) = c;
	}

	inline __host__ __device__ int clip (float3 pos)
	{
		float val = dot(m_normal, pos) - m_d;
		if ( val < 0 )
			return 0;
		else
			return 1;
	}
};

struct plane_clippingPDB
{
	float3		m_normal;
	float		m_d;
	float		m_NormalMag;

	plane_clippingPDB ( float* normal, float D ) // plane defined by Normal and D (distance from origin)
	{
		m_normal 	= make_float3 (normal[0], normal[1], normal[2]);
		m_normal	= normalize (m_normal);
		m_NormalMag	= length (m_normal);
		m_d			= D;
	}

	template <typename Tuple>
	inline __host__ __device__ void operator () (Tuple t)
	{
		float x = thrust::get<0>(t);
		float y = thrust::get<1>(t);
		float z = thrust::get<2>(t);

		int c = clip (make_float3 (x,y,z));

		thrust::get<4>(t) = c;
		thrust::get<5>(t) = c;
		thrust::get<6>(t) = c;
		thrust::get<7>(t) = c;
	}

	inline __host__ __device__ int clip (float3 pos)
	{
		float val = dot(m_normal, pos) - m_d;
		if ( val < 0 )
			return 0;
		else
			return 1;
	}
};

struct plane_clippingPDBver2
{
	float3		m_normal;
	float		m_d;
	float		m_NormalMag;
	
	plane_clippingPDBver2 ( float* normal, float D ) // plane defined by Normal and D (distance from origin)
	{
		m_normal 	= make_float3 (normal[0], normal[1], normal[2]);
		m_normal	= normalize (m_normal);
		m_NormalMag	= length (m_normal);
		m_d			= D;
	}

	template <typename Tuple>
	inline __host__ __device__ bool operator () (Tuple t)
	{
		float x = thrust::get<0>(t);
		float y = thrust::get<1>(t);
		float z = thrust::get<2>(t);

		return clip (make_float3 (x,y,z));

	}

	inline __host__ __device__ int clip (float3 pos)
	{
		float val = dot(m_normal, pos) - m_d;
		if ( val < 0 )
			return 0;
		else
			return 1;
	}
};

template <typename T>
struct not_clipped : public thrust::unary_function<T,bool>
{
    inline __host__ __device__
    bool operator()(T x)
    {
        return x == 0;
    }
};




template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

//forcing it to be a 3-tuple one instead of using variadic templates
template<typename Iterator>
__host__ __device__
thrust::zip_iterator<thrust::tuple<Iterator, Iterator, Iterator, Iterator> > zip(const Iterator& sr1, const Iterator& sr2, const Iterator& sr3, const Iterator& sr4)
{
    return thrust::make_zip_iterator(thrust::make_tuple(sr1, sr2, sr3, sr4 ));
}


#endif /* CCLIPPING_H_ */
