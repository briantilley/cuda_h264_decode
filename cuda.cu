// this entire file was made to test cuda surface mapping
// final code should be properly written, this file may
// need to go

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
CUVIDPROCPARAMS*       pVPP;

void mapSurface( int PicIdx )
{
	unsigned int pitch;

	// made schoolboy error, segfault casued by passing pDecoder
	// fixed below to *pDecoder
	cuvidMapVideoFrame( *pDecoder, PicIdx, &devPtr, &pitch, pVPP );
	cuvidUnmapVideoFrame( *pDecoder, devPtr );
}
