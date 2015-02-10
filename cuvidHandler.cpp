#include <string>
#include <string.h>
#include <iostream>
#include <stdint.h>
#include <unistd.h>

#include "inc/constants.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

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
    // cuvidPicParams->CodecSpecific.h264.CurrFieldOrderCnt[ 0 ] ;
    // cuvidPicParams->CodecSpecific.h264.CurrFieldOrderCnt[ 1 ] ;

    cuvidPicParams->CodecSpecific.h264.fmo_aso_enable = 0;
    cuvidPicParams->CodecSpecific.h264.num_slice_groups_minus1 = PPS.num_slice_groups_minus1;
    cuvidPicParams->CodecSpecific.h264.slice_group_map_type = PPS.slice_group_map_type;
    cuvidPicParams->CodecSpecific.h264.pic_init_qs_minus26 = PPS.pic_init_qs_minus26;
    cuvidPicParams->CodecSpecific.h264.slice_group_change_rate_minus1 = PPS.slice_group_change_rate_minus1;
    // cuvidPicParams->CodecSpecific.h264.fmo.slice_group_map_addr ;

    cuvidPicParams->PicWidthInMbs = SPS.pic_width_in_mbs_minus1;
    // cuvidPicParams->FrameHeightInMbs ;
    // cuvidPicParams->CurrPicIdx ;
    cuvidPicParams->field_pic_flag = SH[ 0 ]->field_pic_flag;
    // cuvidPicParams->bottom_field_flag ;
    // cuvidPicParams->second_field ;

	// cuvidPicParams->nBitstreamDataLen ;
	// cuvidPicParams->pBitstreamData ;
	// cuvidPicParams->nNumSlices ;
	// cuvidPicParams->pSliceDataOffsets ;

	// cuvidPicParams->ref_pic_flag ;
	// cuvidPicParams->intra_pic_flag ;
}

void H264parser::updateDPB( void )
{

}