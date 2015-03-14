// This is the big-daddy GPGPU (image processing) file,
// where all host/device function pairs shall lie.

#include <iostream>
#include <cuda.h>

#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 16

#define cuErrChk(err) { throwGpuError( ( err ), __FILE__, __LINE__ ); }
inline void throwGpuError( cudaError_t error, char* file, int32_t line )
{
	if( cudaSuccess != error )
	{
		fprintf( stderr, "[cuda error] %s:%d %s\n", file, line, cudaGetErrorString( error ) );
		exit( error );
	}
}

__global__ void dNV12toRGBA( const uint8_t*, uint8_t*, const uint32_t, uint32_t, const uint32_t, const uint32_t );
int32_t hNV12toRGBA( const uint8_t*, uint8_t**, const uint32_t, uint32_t*, const uint32_t, const uint32_t );

__global__ void dNV12toBW( const uint8_t*, uint8_t*, const uint32_t, const uint32_t, const uint32_t, const uint32_t );
int32_t hNV12toBW( const uint8_t*, uint8_t**, const uint32_t, uint32_t*, const uint32_t, const uint32_t );

// host-side

int32_t hNV12toRGBA( const uint8_t* dImage, uint8_t** dImageOut,
	const uint32_t pitch, uint32_t* pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	// if( NULL == *dImageOut ) // output not yet allocated
	// 	cuErrChk( cudaMallocPitch( ( void** )dImageOut, ( size_t* )pitchOut, ( size_t )width, ( size_t )height ) );

	dim3 block( BLOCK_WIDTH, BLOCK_HEIGHT );
	dim3 grid( 0, 0 );

	grid.x = ceil( ( float )width / block.x );
	grid.y = ceil( ( float )height / block.y );

	dNV12toRGBA<<< grid, block >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

	cudaDeviceSynchronize( ); cuErrChk( cudaGetLastError( ) );

	return 0;
}

int32_t hNV12toBW( const uint8_t* dImage, uint8_t** dImageOut,
	const uint32_t pitch, uint32_t* pitchOut,
	const uint32_t width,
	const uint32_t height )
{
	if( NULL == dImageOut ) // output not yet allocated
		cuErrChk( cudaMallocPitch( ( void** )dImageOut, ( size_t* )pitchOut, ( size_t )width, ( size_t )height ) );

	dim3 block( BLOCK_WIDTH, BLOCK_HEIGHT );
	dim3 grid( 0, 0 );

	grid.x = ceil( ( float )width / block.x );
	grid.y = ceil( ( float )height / block.y );

	dNV12toBW<<< grid, block >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

	cudaDeviceSynchronize( ); cuErrChk( cudaGetLastError( ) );

	return 0;
}

// host auxiliary functions

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
		uint32_t inIdx  = xIdx /** 4*/ + yIdx * pitch;
		uint32_t outIdx = xIdx * 4 + yIdx * pitchOut;

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

	if( xIdx < 100/*width*/ && yIdx < 100/*height*/ )
	{
		uint32_t inIdx  = xIdx + yIdx * pitch;
		uint32_t outIdx = xIdx + yIdx * pitchOut;

		dImageOut[ outIdx ] = 127;//dImage[ inIdx ];
	}

	return;
}

// device auxiliary functions
