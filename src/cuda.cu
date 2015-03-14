// This is the big-daddy GPGPU (image processing) file,
// where all host/device function pairs shall lie.

#include <iostream>
#include <cuda.h>

#include "inc/constants.h"

#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 16

using std::cout;
using std::endl;
using std::string;

#define cudaErr(err) cudaError( err, __FILE__, __LINE__ )
inline void cudaError( cudaError_t err, const char file[], uint32_t line, bool abort=true )
{
    if( cudaSuccess != err )
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << cudaGetErrorName( err ) << endl;
        if( abort ) exit( err );
    }
}

// device auxiliary functions

// device-side

__global__ void dNV12toRGBA( const uint8_t* dImage, uint8_t* dImageOut,
	const uint32_t pitch, uint32_t pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	pitchOut = ( ( 0 != pitchOut ) ? pitchOut : width * 4 );

	if( xIdx < width && yIdx < height )
	{
		uint32_t inIdx  = xIdx + yIdx * pitch;
		uint32_t outIdx = xIdx * 4 + yIdx * pitchOut;
		// uint32_t pixCount = width * height;

		dImageOut[ outIdx + 0 ] = dImage[ inIdx ];
		dImageOut[ outIdx + 1 ] = dImage[ inIdx ];
		dImageOut[ outIdx + 2 ] = dImage[ inIdx ];
		dImageOut[ outIdx + 3 ] = dImage[ inIdx ];
	}

	return;
}

__global__ void dNV12toBW( const uint8_t* dImage, uint8_t* dImageOut,
	const uint32_t pitch, const uint32_t pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if( xIdx < width && yIdx < height )
	{
		uint32_t inIdx  = xIdx + yIdx * pitch;
		uint32_t outIdx = xIdx + yIdx * pitchOut;

		dImageOut[ outIdx ] = dImage[ inIdx ];
	}

	return;
}

// host auxiliary functions

// host-side

int32_t NV12toRGBA( const uint8_t* dImage, uint8_t** dImageOut,
	const uint32_t pitch, uint32_t* pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	if( NULL == dImage || NULL == *dImageOut) // basic check for image allocation
		cudaErr( cudaErrorInvalidDevicePointer );

	dim3 block( BLOCK_WIDTH, BLOCK_HEIGHT );
	dim3 grid( 0, 0 );

	grid.x = ceil( ( float )width / block.x );
	grid.y = ceil( ( float )height / block.y );

	dNV12toRGBA<<< grid, block >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

	cudaDeviceSynchronize( ); cudaErr( cudaGetLastError( ) );

	return 0;
}

int32_t NV12toBW( const uint8_t* dImage, uint8_t** dImageOut,
	const uint32_t pitch, uint32_t* pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	if( NULL == dImage || NULL == *dImageOut) // basic check for image allocation
		cudaErr( cudaErrorInvalidDevicePointer );

	dim3 block( BLOCK_WIDTH, BLOCK_HEIGHT );
	dim3 grid( 0, 0 );

	grid.x = ceil( ( float )width / block.x );
	grid.y = ceil( ( float )height / block.y );

	dNV12toBW<<< grid, block >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

	cudaDeviceSynchronize( ); cudaErr( cudaGetLastError( ) );

	return 0;
}