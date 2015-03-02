#include <iostream>
#include <cuviddec.h>

#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"
#include "inc/constants.h"

extern CUdevice        dev;
extern CUcontext*      pCtx;
extern CUvideodecoder* pDecoder;

CUdeviceptr            devPtr;

void mapSurface( int PicIdx, CUVIDPROCPARAMS* pVPP)
{
	unsigned int pitch;
std::cout << PicIdx << std::endl;
	// memset( pVPP, 0, sizeof( CUVIDPROCPARAMS ) );
	// pVPP->progressive_frame = 1;
	// pVPP->second_field = 0;
	// pVPP->top_field_first = 0;
	// pVPP->unpaired_field = 1;

	cuvidMapVideoFrame( *pDecoder, PicIdx, &devPtr, &pitch, pVPP );
	cuvidUnmapVideoFrame( *pDecoder, devPtr );
}
