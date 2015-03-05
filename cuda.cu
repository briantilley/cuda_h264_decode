// This is the big daddy GPGPU file,
// where all host/device function pairs shall lie.

#include <iostream>
#include <cuda.h>

__global__ void dNV12toRGBA( const uint8_t*, uint8_t*, const uint32_t, const uint32_t width, const uint32_t, const uint32_t );
int32_t hNV12toRGBA( const uint8_t*, uint8_t**, const uint32_t,	const uint32_t, const uint32_t );

// host-side

int32_t hNV12toRGBA( const uint8_t* dImage, uint8_t** dImageOut,
	const uint32_t pitch,
	const uint32_t width,
	const uint32_t height )
{

	return 0;
}

// device-side

__global__ void dNV12toRGBA( const uint8_t* dImage, uint8_t* dImageOut,
	const uint32_t pitch,
	const uint32_t width,
	const uint32_t height,
	const uint32_t smem )
{

	return;
}

// device auxiliary functions
