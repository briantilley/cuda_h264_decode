#ifndef CUVID_DECODER_H
#define CUVID_DECODER_H

// cuvid dependency
#include <nvcuvid.h>

#define DECODE_SURFACES   8 // higher numbers = more memory usage
#define CLOCK_RATE        0 // default, not sure what this does
#define ERROR_THRESHOLD   10 // tolerate 10% corruption in the video feed
#define DECODE_GAP        2 // number of frames decode should be ahead of map

#define OUTPUT_SURFACES   8 // lower numbers = possible slowdown

class CUVIDparser
{
public:

	// consider using pUserData to differentiate between calling objects
	CUVIDparser( cudaVideoCodec, PFNVIDSEQUENCECALLBACK, PFNVIDDECODECALLBACK, PFNVIDDISPLAYCALLBACK );
	~CUVIDparser( );

	int32_t processPayload( CUvideopacketflags, const uint8_t*, uint64_t, CUvideotimestamp );

private:

	CUvideoparser parser;

};

#endif