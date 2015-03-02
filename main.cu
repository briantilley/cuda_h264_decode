#include <string>
#include <iostream>
#include <stdint.h>
#include <unistd.h>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// create global variables
H264parser parser           = H264parser( );
CUdevice dev;
CUcontext* pCtx             = new CUcontext;
CUvideodecoder* pDecoder    = new CUvideodecoder;
CUVIDDECODECREATEINFO *pdci = new CUVIDDECODECREATEINFO;

// this function is run on data from one frame of video
// essentially, this is the processing callback
int frame_handler( uint8_t* start, uint32_t length )
{
	std::cerr << "." << std::flush;
	
	parser.parseFrame( start, length );
	
	cuvidDecodePicture( *pDecoder, parser.cuvidPicParams );
	
	int mapIdx = parser.idx( ) - 4;
	if( mapIdx >= 0 )
		mapSurface( mapIdx, parser.getProcParams( ) );

	return 0;
}

int main( int argc, char** argv )
{
	// create a V4L2 stream object
	V4L2stream stream = V4L2stream( );
	stream.init( );

	// CUDA code below is ugly and needs to be abstracted
	// create context to appease cuda runtime
	cudaSetDevice( 0 );
	cudaGetDevice( &dev );

	// fill video decoder creation struct
	pdci->ulWidth             = CODED_WIDTH;
	pdci->ulHeight            = CODED_HEIGHT;
	pdci->ulNumDecodeSurfaces = 15;
	pdci->CodecType           = CUVID_CODEC;
	pdci->ChromaFormat        = CUVID_CHROMA;
	pdci->ulCreationFlags     = CUVID_FLAGS;
	pdci->display_area.left   = 0;
	pdci->display_area.top    = 0;
	pdci->display_area.right  = TARGET_WIDTH;
	pdci->display_area.bottom = TARGET_HEIGHT;
	pdci->OutputFormat        = CUVID_OUT_FORMAT;
	pdci->DeinterlaceMode     = CUVID_DEINTERLACE;
	pdci->ulTargetWidth       = TARGET_WIDTH;
	pdci->ulTargetHeight      = TARGET_HEIGHT;
	pdci->ulNumOutputSurfaces = 8;
	pdci->vidLock             = NULL;
	pdci->target_rect.left    = 0;
	pdci->target_rect.top     = 0;
	pdci->target_rect.right   = TARGET_WIDTH;
	pdci->target_rect.bottom  = TARGET_HEIGHT;

	cuvidCreateDecoder( pDecoder, pdci );

	stream.on( );
	for( int i = 0; i < 1200; ++i) // "process" 1200 frames (40 seconds)
		stream.getFrame( &frame_handler );
	stream.off( );

	return 0;
}
