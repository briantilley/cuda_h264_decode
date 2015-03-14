// This is a wrapper class for NVCUVID to simplify decoding and mapping frames

#include <iostream>
#include <cstring>

#include "inc/CUVIDparser.h"

using std::cout;
using std::endl;
using std::string;

// cuda driver error checking
#define cuErr(err) cuError( err, __FILE__, __LINE__ )
inline void cuError( CUresult err, const char file[], uint32_t line, bool abort=true )
{
    if( CUDA_SUCCESS != err )
    {
    	const char* str;
    	cuGetErrorName( err, &str );

        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << str << endl;
        if( abort ) exit( err );
    }
}

// make a simple cuvid parser object in the constructor
// create a decoder upon first parse
CUVIDparser::CUVIDparser( cudaVideoCodec cudaCodec, PFNVIDSEQUENCECALLBACK fn_vidSeq, PFNVIDDECODECALLBACK fn_vidDec, PFNVIDDISPLAYCALLBACK fn_vidDisp )
{
	// create and initialize a params struct to make a CUVID parser object
	CUVIDPARSERPARAMS params;

	params.CodecType              = cudaCodec;
	params.ulMaxNumDecodeSurfaces = DECODE_SURFACES;
	params.ulClockRate            = CLOCK_RATE;
	params.ulErrorThreshold       = ERROR_THRESHOLD;
	params.ulMaxDisplayDelay      = DECODE_GAP;
	
	memset( params.uReserved1, 0, sizeof( params.uReserved1 ) );

	params.pUserData              = NULL; // not currently in use
	params.pfnSequenceCallback    = fn_vidSeq;
	params.pfnDecodePicture       = fn_vidDec;
	params.pfnDisplayPicture      = fn_vidDisp;

	memset( params.pvReserved2, 0, sizeof( params.pvReserved2 ) );

	params.pExtVideoInfo          = NULL; // not currently in use

	// make the aforementioned parser object (CUVIDparser class data member)
	cuErr( cuvidCreateVideoParser( &parser, &params ) );
}

// clean up allocated memory
CUVIDparser::~CUVIDparser( void )
{
	// destroy parser object
	cuErr( cuvidDestroyVideoParser( parser ) );
}

// parse payload
int32_t CUVIDparser::processPayload( CUvideopacketflags cuvidPktFlags, const uint8_t* in_payload, uint64_t payloadSize, CUvideotimestamp in_timestamp )
{
	// make a source data packet given the arguments
	CUVIDSOURCEDATAPACKET sdp;

	sdp.flags        = cuvidPktFlags;
	sdp.payload_size = payloadSize;
	sdp.payload      = in_payload;
	sdp.timestamp    = in_timestamp;

	// parse the data packet
	// this function will launch the sequence, decode, and display
	// callbacks as necessary
	cuErr( cuvidParseVideoData( parser, &sdp ) );
}