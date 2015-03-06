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

__global__ void dNV12toRGBA( const uint8_t*, uint8_t*, const uint32_t, const uint32_t, const uint32_t, const uint32_t );
int32_t hNV12toRGBA( const uint8_t*, uint8_t**, const uint32_t,	const uint32_t, const uint32_t );

// host-side

int32_t hNV12toRGBA( const uint8_t* dImage, uint8_t** dImageOut,
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

	dNV12toRGBA<<< grid, block >>>( dImage, *dImageOut, pitch, *pitchOut, width, height );

	cudaDeviceSynchronize( ); cuErrChk( cudaGetLastError( ) );

	return 0;
}

// host auxiliary functions

// device-side

__global__ void dNV12toRGBA( const uint8_t* dImage, uint8_t* dImageOut,
	const uint32_t pitch, const uint32_t pitchOut,
	const uint32_t width,
	const uint32_t height )
{

	return;
}

// device auxiliary functions
