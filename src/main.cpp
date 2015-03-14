// Goal of this application is to

// 1. retrieve an H264-encoded frame of video from a webcam with V4L2
// 2. parse the encoded frame with NVCUVID
// 3. decode the encoded frame on the GPU with NVCUVID
// 4. process the decoded frame on the GPU with CUDA
// 5. display the processed frame with openGL

// Using this program flow, raw image data originates on and is not
// copied via the PCIe bus or USB, which would cause the program
// to spend most of its time waiting for data to transfer.

#include <iostream>
#include <fstream>
#include <cstring>
#include <nvcuvid.h>

#include "inc/V4L2stream.h"
#include "inc/CUVIDparser.h"
#include "inc/GLviewer.h"
#include "inc/constants.h"
#include "inc/functions.h"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;

// create global variables
V4L2stream* stream;
CUVIDparser* decoder;
GLviewer* viewer;
CUvideodecoder cuDecoder;

CUdevice dev;

uint32_t width, height;

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

// end the application
void appEnd( void )
{
    cout << endl;

    stream->off( );
    delete stream;
    delete decoder;
    delete viewer;
    exit( 0 );
}

// just as the function name indicates
string loadTxtFileAsString( const char shaderFileName[] )
{
    string source;
    string buf = "";
    ifstream file( shaderFileName, std::ios::in );

    while( file.good( ) )
    {
        std::getline( file, buf );
        source.append( buf + "\n" );
    }

    file.close( );

    return source;
}

// callback for coded frame input
int32_t input_callback( uint8_t* start, uint32_t length )
{
    cout << "." << std::flush;

    decoder->processPayload( ( CUvideopacketflags )0, ( const uint8_t* )start, length, 0 );

    return 0;
}

// three callbacks below go to cuvid parser
int seq_callback( void *pUserData, CUVIDEOFORMAT* pVidFmt )
{
    // make the decoder if necessary
    if( NULL == cuDecoder )
    {
        // fill creation struct with pVidFmt
        CUVIDDECODECREATEINFO dci;

        dci.ulWidth             = pVidFmt->coded_width;
        dci.ulHeight            = pVidFmt->coded_height;
        dci.ulNumDecodeSurfaces = DECODE_SURFACES;
        dci.CodecType           = cudaVideoCodec_H264; // magic value
        dci.ChromaFormat        = cudaVideoChromaFormat_422; // magic value
        dci.ulCreationFlags     = cudaVideoCreate_Default; // magic value

        memset( dci.Reserved1, 0, sizeof( dci.Reserved1 ) );

        dci.display_area.left   = pVidFmt->display_area.left;
        dci.display_area.top    = pVidFmt->display_area.top;
        dci.display_area.right  = pVidFmt->display_area.right;
        dci.display_area.bottom = pVidFmt->display_area.bottom;

        dci.OutputFormat        = cudaVideoSurfaceFormat_NV12; // only fmt supported
        dci.DeinterlaceMode     = cudaVideoDeinterlaceMode_Adaptive; // magic value
        dci.ulTargetWidth       = dci.display_area.right;
        dci.ulTargetHeight      = dci.display_area.bottom;
        dci.ulNumOutputSurfaces = OUTPUT_SURFACES;
        dci.vidLock             = NULL; // come back to this for multithreading
        dci.target_rect.left    = 0;
        dci.target_rect.top     = 0;
        dci.target_rect.right   = dci.ulTargetWidth;
        dci.target_rect.bottom  = dci.ulTargetHeight;

        memset( dci.Reserved2, 0, sizeof( dci.Reserved2 ) );

        // create the decoder
        cuErr( cuvidCreateDecoder( &cuDecoder, &dci ) );
    }

    return 1; // unfortunately, 1 is no error and 0 is error
}

int dec_callback( void *pUserData, CUVIDPICPARAMS* pPicParams )
{
    // fatal error if decoder is empty
    if( 0 == cuDecoder ) { cout << "decoder empty" << endl; return 0; }

    // decode picture
    cuErr( cuvidDecodePicture( cuDecoder, pPicParams ) );

    return 1; // unfortunately, 1 is no error and 0 is error
}

int disp_callback( void *pUserData, CUVIDPARSERDISPINFO* pParDispInfo )
{
    // manage values for the cuvid display image
    int32_t PicIdx = pParDispInfo->picture_index;
    CUdeviceptr src;
    uint32_t srcPitch;
    CUVIDPROCPARAMS trashVPP;

    // manage values for the cuda-gl output image
    uint8_t* dest = NULL;
    uint32_t destPitch = 0;

    // map the cuvid-frame
    cuErr( cuvidMapVideoFrame( cuDecoder, PicIdx, &src, &srcPitch, &trashVPP ) );

    // map the cuda-gl frame
    // process & copy cuvid frame to cuda-gl
    // unmap cuda-gl frame
    viewer->mapDispImage( ( void** )&dest/*, &pitch*/ );
    NV12toRGBA( ( uint8_t* )src, &dest, srcPitch, &destPitch, width, height );
    viewer->unmapDispImage( );

    // display cuda-gl image
    viewer->display( );

    // unmap cuvid image
    cuErr( cuvidUnmapVideoFrame( cuDecoder, src ) );

    return 1; // unfortunately, 1 is no error and 0 is error
}

// get a payload from v4l2
// parse the payload with cuvid parser
// decode with cuvid decoder
// post-process (convert) with CUDA
// display with openGL
int main( int argc, const char* argv[] )
{
    // variables for width and height to use programatically
    width = WIDTH;
    height = HEIGHT;

    cuDecoder = NULL;

	// create a V4L2stream object
	stream = new V4L2stream( );
	stream->init( &width, &height, DEVICE, INPUT_SURFACES );

    // create a CUVIDparser object
    decoder = new CUVIDparser( cudaVideoCodec_H264, seq_callback, dec_callback, disp_callback );

    // create a GLviewer object
    viewer = new GLviewer( width, height, 1920, 1080, GLviewer_fullscreen | GLviewer_color, appEnd );

	stream->on( );
    //for( int i = 0; i < 1800; ++i) // 60 seconds
	while(true)
    {
    	stream->getCodedFrame( input_callback );
        viewer->loop( );
    }

	appEnd( );

	return 0;
}