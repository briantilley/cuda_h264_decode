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
	extern __shared__ uint32_t smem[];

	uint32_t xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	pitchOut = ( ( 0 != pitchOut ) ? pitchOut : width * 4 );

	if( xIdx < width && yIdx < height ) // keep inside image
	{
		uint32_t inIdx  = xIdx + yIdx * pitch;
		uint32_t outIdx = xIdx * 4 + yIdx * pitchOut;

		if( threadIdx.y < blockDim.y / 2 ) // only need first half of block to load Cb & Cr
		{
			// make s_inIdx and smemIdx to copy from global to shared
			uint32_t tempY = height + threadIdx.y + ( blockDim.y / 2 ) * blockIdx.y;

			uint32_t s_inIdx = xIdx + tempY * pitch;
			uint32_t smemIdx = threadIdx.x + threadIdx.y * blockDim.x;

			smem[ smemIdx ] = dImage[ s_inIdx ];
		}

		__syncthreads( ); // shared memory usage is prone to data

		// make smemIdx to read UV from shared
		uint32_t tempY = threadIdx.y / 2;
		uint32_t tempX = threadIdx.x - ( threadIdx.x % 2 );

		uint32_t smemIdx = tempX + tempY * blockDim.x;

		// expecting lots of bank conflicts, can't do much about it
		float Y = dImage[ inIdx ];
		float U = smem[ smemIdx ];
		float V = smem[ smemIdx + 1 ];

		// // global read, likely slower
		// float Y = dImage[ inIdx ];
		// float U = dImage[ pitch * ( height + yIdx / 2 ) + xIdx - ( xIdx % 2 ) ];
		// float V = dImage[ pitch * ( height + yIdx / 2 ) + xIdx - ( xIdx % 2 ) + 1 ];

		// YUV to RGB math
		Y -= 16;
		U -= 128;
		V -= 128;

		Y *= 255;
		U *= 127;
		V *= 127;

		Y /= 219;
		U /= 112;
		V /= 112;

		// simultaneously clamp and convert
		dImageOut[ outIdx + 0 ] = max( min( ( int32_t )( Y + 1.402f * V + 0.5f ), 255 ), 0 );
		dImageOut[ outIdx + 1 ] = max( min( ( int32_t )( Y - 0.344f * U - 0.714f * V + 0.5f ), 255 ), 0 );
		dImageOut[ outIdx + 2 ] = max( min( ( int32_t )( Y + 1.722f * U + 0.5f ), 255 ), 0 );
		dImageOut[ outIdx + 3 ] = 255;
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

	uint32_t smem = ( block.x + 1 ) * block.y / 2; // shared memory allocation makes this more efficient
	smem *= sizeof( uint32_t );

	dNV12toRGBA<<< grid, block, smem >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

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