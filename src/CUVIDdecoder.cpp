// This is a wrapper class for NVCUVID to simplify decoding and mapping frames

#include <iostream>

#include "inc/constants.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// cuda driver error checking
#define cuErr(err) cuError( err, __FILE__, __LINE__ )
inline void cuError( CUresult err, const char* file, uint32_t line, bool abort=true )
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
CUVIDdecoder::CUVIDdecoder( cudaVideoCodec cudaCodec, PFNVIDSEQUENCECALLBACK fn_vidSeq, PFNVIDDECODECALLBACK fn_vidDec, PFNVIDDISPLAYCALLBACK fn_vidDisp )
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

	// make the aforementioned parser object (CUVIDdecoder class data member)
	cuErr( cuvidCreateVideoParser( &parser, &params ) );

	// initialize decoder data member to NULL
	decoder = 0;
}

// clean up allocated memory
CUVIDdecoder::~CUVIDdecoder( void )
{
	// destroy parser and decoder objects
	cuErr( cuvidDestroyVideoParser( parser ) );
	cuErr( cuvidDestroyDecoder( decoder ) );
}

// parse payload
int32_t CUVIDdecoder::processPayload( CUvideopacketflags cuvidPktFlags, const uint8_t* in_payload, uint64_t payloadSize, CUvideotimestamp in_timestamp )
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

/*
CUVIDdecoder::CUVIDdecoder( uint32_t width,
	uint32_t height,
	CUVIDdecoder_fmt format )
{
	pdci = new CUVIDDECODECREATEINFO;

	// coded width and height are rounded up to the nearest multiple of 16
	pdci->ulWidth             = width  + 16 - ( width  % 16 );
	pdci->ulHeight            = height + 16 - ( height % 16 );
	pdci->ulNumDecodeSurfaces = DECODE_SURFACES;

	switch( format )
	{
		case CUVIDdecoder_NULL:
			std::cerr << "invalid decoder format" << endl;
		break;
		case CUVIDdecoder_H264:
			pdci->CodecType   = cudaVideoCodec_H264;
		break;
		default:
			std::cerr << "decoder format not recognized" << endl;
		break;
	}

	pdci->ChromaFormat        = CUVID_CHROMA;
	pdci->ulCreationFlags     = CUVID_FLAGS;
	pdci->display_area.left   = 0;
	pdci->display_area.top    = 0;
	pdci->display_area.right  = width;
	pdci->display_area.bottom = height;
	pdci->OutputFormat        = CUVID_OUT_FORMAT;
	pdci->DeinterlaceMode     = CUVID_DEINTERLACE;
	pdci->ulTargetWidth       = width;
	pdci->ulTargetHeight      = height;
	pdci->ulNumOutputSurfaces = OUTPUT_SURFACES;
	pdci->vidLock             = NULL;
	pdci->target_rect.left    = 0;
	pdci->target_rect.top     = 0;
	pdci->target_rect.right   = width;
	pdci->target_rect.bottom  = height;

	cuErr( cuvidCreateDecoder( &decoder, pdci ) );
}
*/

/*
CUVIDdecoder::~CUVIDdecoder( void )
{
	delete pdci;
	cuErr( cuvidDestroyDecoder( decoder ) );
}
*/

/*
int32_t CUVIDdecoder::getDecodedFrame( CUVIDPICPARAMS* picParams, int32_t ( * cuda_callback )( const CUdeviceptr, uint32_t pitch ) )
{
	// our data
	const int32_t idx = picParams->CurrPicIdx;
	CUVIDPROCPARAMS     junkVPP; // not needed here, required by cuvid

	// callback's data
	CUdeviceptr devPtr;
	uint32_t    pitch;

	// decode frame
	cuErr( cuvidDecodePicture( decoder, picParams ) );

	// make sure there's a gap between decode and map
	if( idx < DECODE_GAP )
		return 2; // decoder not far enough ahead (non-fatal)

	// map frame
	cuErr( cuvidMapVideoFrame( decoder, idx - DECODE_GAP, &devPtr, &pitch, &junkVPP ) );

	// run callback
	int32_t err = cuda_callback( devPtr, pitch );
	if( 0 != err ) std::cerr << "error " << err << " in cuda_callback" << endl;

	// unmap frame
	cuErr( cuvidUnmapVideoFrame( decoder, devPtr ) );

	return 0;
}
*/