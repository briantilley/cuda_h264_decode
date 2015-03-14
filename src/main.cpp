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

#include "inc/constants.h"
#include "inc/classes.h"
#include "inc/functions.h"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;

// create global variables
V4L2stream* stream;
// H264parser* parser;
// CUVIDPICPARAMS* picParams;
CUVIDdecoder* decoder;
GLviewer* viewer;
CUvideodecoder cuDecoder;

CUdevice dev;

uint32_t width, height;

// cuda driver error checking
#define cuErr(err) cuError( err, __FILE__, __LINE__ )
inline void cuError( CUresult err, const char* file, uint32_t line, bool abort=true )
{
    if( CUDA_SUCCESS != err )
    {
        std::cerr << "caugt an error" << endl;
        const char* str;
        cuGetErrorName( err, &str );

        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << str << endl;
        if( abort ) exit( err );
    }
}

// callback for coded frame input
int32_t input_callback( uint8_t* start, uint32_t length )
{
    cout << "." << std::flush;

    decoder->processPayload( ( CUvideopacketflags )0, ( const uint8_t* )start, length, 0 );

    /*
    std::cerr << "." << std::flush;
    
    parser->parseFrame( start, length );

    fillCuvidPicParams( parser, picParams );
    updateCuvidDPB( parser, picParams );

    decoder->getDecodedFrame( picParams, decoded_callback );
    */

    return 0;
}

/*
// callback for decoded frame input
int32_t decoded_callback( const CUdeviceptr devPtr, uint32_t pitch )
{
    const uint8_t* src = ( const uint8_t* )devPtr;
    uint8_t* dest = NULL;
    uint32_t pitchOut = 0;

    viewer->mapOutputImage( &dest );
    hNV12toRGBA( src, &dest, pitch, &pitchOut, WIDTH, HEIGHT );
    viewer->unmapOutputImage( );

    viewer->display( );

	return 0;
} */

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
    int32_t PicIdx = pParDispInfo->picture_index;
    CUdeviceptr devPtr;
    uint32_t pitch;
    CUVIDPROCPARAMS trashVPP;

    cuErr( cuvidMapVideoFrame( cuDecoder, PicIdx, &devPtr, &pitch, &trashVPP ) );
    cuErr( cuvidUnmapVideoFrame( cuDecoder, devPtr ) );

    return 1; // unfortunately, 1 is no error and 0 is error
}

int main( int argc, char** argv )
{
    // variables for width and height to use programatically
    width = WIDTH;
    height = HEIGHT;

    cuDecoder = NULL;

	// create a V4L2stream object
	stream = new V4L2stream( );
	stream->init( &width, &height, DEVICE, INPUT_SURFACES );

	// // create an H264parser object
    // // using CUVID parser in this git branch
	// parser = new H264parser( BitPos( NULL, 0 ) );

	// initializing CUVIDPICPARAMS, consider putting struct in decoder class
	// picParams = new CUVIDPICPARAMS;
	// picParams->CurrPicIdx = -1;
	// clearCuvidDPB( picParams );

    // create a CUVIDdecoder object
    decoder = new CUVIDdecoder( cudaVideoCodec_H264, seq_callback, dec_callback, disp_callback );
    // decoder = new CUVIDdecoder( WIDTH, HEIGHT, CUVIDdecoder_H264 );

    // create a GLviewer object
    viewer = new GLviewer( width, height, GLcolor_RGBA );

	stream->on( );
	/*while(true)*/for( int i = 0; i < 1800; ++i) // 60 seconds
    {
    	stream->getCodedFrame( &input_callback );
    }

	cout << endl;

	cleanUp( );

	return 0;
}

// auxiliary functions

void cleanUp( void )
{
    stream->off( );
    delete stream;
    // delete parser;
	// delete picParams;
    delete decoder;
    delete viewer;
}

string loadTxtFileAsString( const char* shaderFileName )
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

/*
int fillCuvidPicParams( H264parser* parser, CUVIDPICPARAMS* params )
{
	params->CodecSpecific.h264.log2_max_frame_num_minus4 = parser->SPS.log2_max_frame_num_minus4;
	params->CodecSpecific.h264.pic_order_cnt_type = parser->SPS.pic_order_cnt_type;
	params->CodecSpecific.h264.log2_max_pic_order_cnt_lsb_minus4 = parser->SPS.log2_max_pic_order_cnt_lsb_minus4;
	params->CodecSpecific.h264.delta_pic_order_always_zero_flag = parser->SPS.delta_pic_order_always_zero_flag;
	params->CodecSpecific.h264.frame_mbs_only_flag = parser->SPS.frame_mbs_only_flag;
	params->CodecSpecific.h264.direct_8x8_inference_flag = parser->SPS.direct_8x8_inference_flag;
	params->CodecSpecific.h264.num_ref_frames = parser->SPS.max_num_ref_frames;
	params->CodecSpecific.h264.residual_colour_transform_flag = 0; //----------------
	params->CodecSpecific.h264.bit_depth_luma_minus8 = parser->SPS.bit_depth_luma_minus8;
	params->CodecSpecific.h264.bit_depth_chroma_minus8 = parser->SPS.bit_depth_chroma_minus8;
	params->CodecSpecific.h264.qpprime_y_zero_transform_bypass_flag = parser->SPS.qpprime_y_zero_transform_bypass_flag;
    params->CodecSpecific.h264.entropy_coding_mode_flag = parser->PPS.entropy_coding_mode_flag;
    params->CodecSpecific.h264.pic_order_present_flag = parser->PPS.bottom_field_pic_order_in_frame_present_flag;
    params->CodecSpecific.h264.num_ref_idx_l0_active_minus1 = parser->PPS.num_ref_idx_l0_default_active_minus1;
    params->CodecSpecific.h264.num_ref_idx_l1_active_minus1 = parser->PPS.num_ref_idx_l1_default_active_minus1;
    params->CodecSpecific.h264.weighted_pred_flag = parser->PPS.weighted_pred_flag;
    params->CodecSpecific.h264.weighted_bipred_idc = parser->PPS.weighted_bipred_idc;
    params->CodecSpecific.h264.pic_init_qp_minus26 = parser->PPS.pic_init_qp_minus26;
    params->CodecSpecific.h264.deblocking_filter_control_present_flag = parser->PPS.deblocking_filter_control_present_flag;
    params->CodecSpecific.h264.redundant_pic_cnt_present_flag = parser->PPS.redundant_pic_cnt_present_flag;
    params->CodecSpecific.h264.transform_8x8_mode_flag = parser->PPS.transform_8x8_mode_flag;
    params->CodecSpecific.h264.MbaffFrameFlag = parser->SPS.mb_adaptive_frame_field_flag && !parser->SH[ 0 ]->field_pic_flag;
    params->CodecSpecific.h264.constrained_intra_pred_flag = parser->PPS.constrained_intra_pred_flag;
    params->CodecSpecific.h264.chroma_qp_index_offset = parser->PPS.chroma_qp_index_offset;
    params->CodecSpecific.h264.second_chroma_qp_index_offset = parser->PPS.second_chroma_qp_index_offset;
    params->CodecSpecific.h264.ref_pic_flag = ( parser->nal_ref_idc ) ? 1 : 0;
    params->CodecSpecific.h264.frame_num = parser->SH[ 0 ]->frame_num;

    for( int i = 0; i < 6; ++i )
    	for( int j = 0; j < 16; ++j )
    		params->CodecSpecific.h264.WeightScale4x4[i][j] = parser->weightScale4x4[i][j];

    for( int i = 0; i < 2; ++i )
    	for( int j = 0; j < 64; ++j )
    		params->CodecSpecific.h264.WeightScale8x8[i][j] = parser->weightScale8x8[i][j];

    params->CodecSpecific.h264.fmo_aso_enable = 0; //--------------------------------
    params->CodecSpecific.h264.num_slice_groups_minus1 = parser->PPS.num_slice_groups_minus1;
    params->CodecSpecific.h264.slice_group_map_type = parser->PPS.slice_group_map_type;
    params->CodecSpecific.h264.pic_init_qs_minus26 = parser->PPS.pic_init_qs_minus26;
    params->CodecSpecific.h264.slice_group_change_rate_minus1 = parser->PPS.slice_group_change_rate_minus1;
    // params->CodecSpecific.h264.fmo.slice_group_map_addr ;

    params->PicWidthInMbs = parser->SPS.pic_width_in_mbs_minus1 + 1;
    params->FrameHeightInMbs = ( 2 - parser->SPS.frame_mbs_only_flag ) * ( parser->SPS.pic_height_in_map_units_minus1 + 1 );
    ++params->CurrPicIdx;
    params->field_pic_flag = parser->SH[ 0 ]->field_pic_flag;
    params->bottom_field_flag = parser->SH[ 0 ]->bottom_field_flag;
    params->second_field = ( parser->PrevFrameNum == parser->SH[ 0 ]->frame_num && parser->SH[ 0 ]->field_pic_flag ) ? 1 : 0;

	params->nBitstreamDataLen = parser->length;
	params->pBitstreamData = parser->start;
	params->nNumSlices = parser->SHidx;
	params->pSliceDataOffsets = parser->SDOs;

	params->ref_pic_flag = ( parser->nal_ref_idc ) ? 1 : 0;
	params->intra_pic_flag = parser->idr_pic_flag;

	return 0;
}

int updateCuvidDPB( H264parser* parser, CUVIDPICPARAMS* params )
{
	CUVIDH264DPBENTRY* dpb = params->CodecSpecific.h264.dpb;

    if( params->intra_pic_flag )
    {
        clearCuvidDPB( params );
        dpb[ 0 ].PicIdx = params->CurrPicIdx;
        dpb[ 0 ].FrameIdx = ( parser->SH[ 0 ]->pDRPM->long_term_reference_flag ) ? parser->SH[ 0 ]->pDRPM->long_term_frame_idx : parser->SH[ 0 ]->frame_num;
        dpb[ 0 ].is_long_term = parser->SH[ 0 ]->pDRPM->long_term_reference_flag;
        dpb[ 0 ].not_existing = 0;
        dpb[ 0 ].used_for_reference = 1;
    }

    for ( int i = 0; i < DPB_SIZE; ++i )
    {
        if( 1 == dpb[ i ].not_existing && 0 == dpb[ i ].used_for_reference )
        {
            if( params->second_field ) return 0; // would be redundant to store second field

            if( params->ref_pic_flag ) // only store ref pics
            {
                dpb[ i ].PicIdx = params->CurrPicIdx;
                dpb[ i ].FrameIdx = ( parser->SH[ 0 ]->pDRPM->long_term_reference_flag ) ? parser->SH[ 0 ]->pDRPM->long_term_frame_idx : parser->SH[ 0 ]->frame_num;
                dpb[ i ].is_long_term = parser->SH[ 0 ]->pDRPM->long_term_reference_flag;
                dpb[ i ].not_existing = 0;
                dpb[ i ].used_for_reference = 1;
            }

            return 0; // store one and done
        }
    }

	return 0;
}

int clearCuvidDPB( CUVIDPICPARAMS* params )
{
    for( int i = 0; i < DPB_SIZE; ++i )
    {
        params->CodecSpecific.h264.dpb[ i ].PicIdx = -1;
        params->CodecSpecific.h264.dpb[ i ].not_existing = 1;
        params->CodecSpecific.h264.dpb[ i ].used_for_reference = 0;
    }

    return 0;
}*/