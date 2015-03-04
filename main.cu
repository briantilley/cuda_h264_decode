#include <iostream>

#include "inc/constants.h"
#include "inc/classes.h"
#include "inc/functions.h"

using std::cout;
using std::endl;
using std::string;

// create global variables
H264parser parser           = H264parser( BitPos( NULL, 0 ) );
CUVIDPICPARAMS* picParams   = new CUVIDPICPARAMS;
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

	fillCuvidPicParams( &parser, picParams );
	updateCuvidDPB( &parser, picParams );

	cuvidDecodePicture( *pDecoder, picParams );

	return 0;
}

int main( int argc, char** argv )
{
	// create a V4L2 stream object
	V4L2stream stream = V4L2stream( TARGET_WIDTH, TARGET_HEIGHT, "/dev/video0", 8 );
	stream.init( );

	// initializing CUVIDPICPARAMS, consider putting struct in decoder class
	picParams->CurrPicIdx = -1;
	clearCuvidDPB( picParams );

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

	cout << endl;

	cleanUp( );

	return 0;
}

void cleanUp( void )
{
	delete picParams;
	delete pCtx;
	delete pDecoder;
	delete pdci;
}

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
}