#include <iostream>

#include "inc/constants.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// straightforward population of the pic params struct for cuda decoder
// some values are hardcoded magic numbers, not a major issue here
void H264parser::fillParams( )
{
	cuvidPicParams->CodecSpecific.h264.log2_max_frame_num_minus4 = SPS.log2_max_frame_num_minus4;
	cuvidPicParams->CodecSpecific.h264.pic_order_cnt_type = SPS.pic_order_cnt_type;
	cuvidPicParams->CodecSpecific.h264.log2_max_pic_order_cnt_lsb_minus4 = SPS.log2_max_pic_order_cnt_lsb_minus4;
	cuvidPicParams->CodecSpecific.h264.delta_pic_order_always_zero_flag = SPS.delta_pic_order_always_zero_flag;
	cuvidPicParams->CodecSpecific.h264.frame_mbs_only_flag = SPS.frame_mbs_only_flag;
	cuvidPicParams->CodecSpecific.h264.direct_8x8_inference_flag = SPS.direct_8x8_inference_flag;
	cuvidPicParams->CodecSpecific.h264.num_ref_frames = SPS.max_num_ref_frames;
	cuvidPicParams->CodecSpecific.h264.residual_colour_transform_flag = 0; //----------------
	cuvidPicParams->CodecSpecific.h264.bit_depth_luma_minus8 = SPS.bit_depth_luma_minus8;
	cuvidPicParams->CodecSpecific.h264.bit_depth_chroma_minus8 = SPS.bit_depth_chroma_minus8;
	cuvidPicParams->CodecSpecific.h264.qpprime_y_zero_transform_bypass_flag = SPS.qpprime_y_zero_transform_bypass_flag;
    cuvidPicParams->CodecSpecific.h264.entropy_coding_mode_flag = PPS.entropy_coding_mode_flag;
    cuvidPicParams->CodecSpecific.h264.pic_order_present_flag = PPS.bottom_field_pic_order_in_frame_present_flag;
    cuvidPicParams->CodecSpecific.h264.num_ref_idx_l0_active_minus1 = PPS.num_ref_idx_l0_default_active_minus1;
    cuvidPicParams->CodecSpecific.h264.num_ref_idx_l1_active_minus1 = PPS.num_ref_idx_l1_default_active_minus1;
    cuvidPicParams->CodecSpecific.h264.weighted_pred_flag = PPS.weighted_pred_flag;
    cuvidPicParams->CodecSpecific.h264.weighted_bipred_idc = PPS.weighted_bipred_idc;
    cuvidPicParams->CodecSpecific.h264.pic_init_qp_minus26 = PPS.pic_init_qp_minus26;
    cuvidPicParams->CodecSpecific.h264.deblocking_filter_control_present_flag = PPS.deblocking_filter_control_present_flag;
    cuvidPicParams->CodecSpecific.h264.redundant_pic_cnt_present_flag = PPS.redundant_pic_cnt_present_flag;
    cuvidPicParams->CodecSpecific.h264.transform_8x8_mode_flag = PPS.transform_8x8_mode_flag;
    cuvidPicParams->CodecSpecific.h264.MbaffFrameFlag = SPS.mb_adaptive_frame_field_flag && !SH[ 0 ]->field_pic_flag;
    cuvidPicParams->CodecSpecific.h264.constrained_intra_pred_flag = PPS.constrained_intra_pred_flag;
    cuvidPicParams->CodecSpecific.h264.chroma_qp_index_offset = PPS.chroma_qp_index_offset;
    cuvidPicParams->CodecSpecific.h264.second_chroma_qp_index_offset = PPS.second_chroma_qp_index_offset;
    cuvidPicParams->CodecSpecific.h264.ref_pic_flag = ( nal_ref_idc ) ? 1 : 0;
    cuvidPicParams->CodecSpecific.h264.frame_num = SH[ 0 ]->frame_num;

    cuvidPicParams->CodecSpecific.h264.fmo_aso_enable = 0; //--------------------------------
    cuvidPicParams->CodecSpecific.h264.num_slice_groups_minus1 = PPS.num_slice_groups_minus1;
    cuvidPicParams->CodecSpecific.h264.slice_group_map_type = PPS.slice_group_map_type;
    cuvidPicParams->CodecSpecific.h264.pic_init_qs_minus26 = PPS.pic_init_qs_minus26;
    cuvidPicParams->CodecSpecific.h264.slice_group_change_rate_minus1 = PPS.slice_group_change_rate_minus1;
    // cuvidPicParams->CodecSpecific.h264.fmo.slice_group_map_addr ;

    cuvidPicParams->PicWidthInMbs = SPS.pic_width_in_mbs_minus1 + 1;
    cuvidPicParams->FrameHeightInMbs = ( 2 - SPS.frame_mbs_only_flag ) * ( SPS.pic_height_in_map_units_minus1 + 1 );
    // cuvidPicParams->CurrPicIdx = ( cuvidPicParams->intra_pic_flag ) ? 0 : cuvidPicParams->CurrPicIdx; // reset the index at every IDR pic
    cuvidPicParams->field_pic_flag = SH[ 0 ]->field_pic_flag;
    cuvidPicParams->bottom_field_flag = SH[ 0 ]->bottom_field_flag;
    // second_field handled in H264parser::parseFrame switch statement

	cuvidPicParams->nBitstreamDataLen = length;
	cuvidPicParams->pBitstreamData = start;
	cuvidPicParams->nNumSlices = SHidx;
	cuvidPicParams->pSliceDataOffsets = SDOs;

	cuvidPicParams->ref_pic_flag = ( nal_ref_idc ) ? 1 : 0;
	cuvidPicParams->intra_pic_flag = idr_pic_flag;

    updateDPB( ); // manage the decoded picture buffer
}

// functions for managing the decoded picture buffer
// behavior is specified by the H264 standard
void H264parser::updateDPB( void )
{
    CUVIDH264DPBENTRY* dpb = cuvidPicParams->CodecSpecific.h264.dpb;

    if( cuvidPicParams->intra_pic_flag )
    {
        clearDPB( );
        dpb[ 0 ].PicIdx = cuvidPicParams->CurrPicIdx;
        dpb[ 0 ].FrameIdx = ( SH[ 0 ]->pDRPM->long_term_reference_flag ) ? SH[ 0 ]->pDRPM->long_term_frame_idx : SH[ 0 ]->frame_num;
        dpb[ 0 ].is_long_term = SH[ 0 ]->pDRPM->long_term_reference_flag;
        dpb[ 0 ].not_existing = 0;
        dpb[ 0 ].used_for_reference = 1;
    }

    for ( int i = 0; i < DPB_SIZE; ++i )
    {
        if( 1 == dpb[ i ].not_existing && 0 == dpb[ i ].used_for_reference )
        {
            if( cuvidPicParams->second_field ) return; // would be redundant to store second field

            if( cuvidPicParams->ref_pic_flag ) // only store ref pics
            {
                dpb[ i ].PicIdx = cuvidPicParams->CurrPicIdx;
                dpb[ i ].FrameIdx = ( SH[ 0 ]->pDRPM->long_term_reference_flag ) ? SH[ 0 ]->pDRPM->long_term_frame_idx : SH[ 0 ]->frame_num;
                dpb[ i ].is_long_term = SH[ 0 ]->pDRPM->long_term_reference_flag;
                dpb[ i ].not_existing = 0;
                dpb[ i ].used_for_reference = 1;
            }

            return; // store one and done
        }
    }
}

void H264parser::clearDPB( void )
{
    for( int i = 0; i < DPB_SIZE; ++i )
    {
        cuvidPicParams->CodecSpecific.h264.dpb[ i ].PicIdx = -1;
        cuvidPicParams->CodecSpecific.h264.dpb[ i ].not_existing = 1;
        cuvidPicParams->CodecSpecific.h264.dpb[ i ].used_for_reference = 0;
    }
}
