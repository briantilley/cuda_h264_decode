#include <string>
#include <iostream>
#include <nvcuvid.h>
#include <cmath>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// using std::ceil;
// using std::log2;

void H264parser::seqPmSet( uint8_t nal_ref_idc, uint8_t nal_type )
{
	SPS.profile_idc                              = uv( 8 );
	SPS.constraint_set0_flag                     = uv( 1 );
	SPS.constraint_set1_flag                     = uv( 1 );
	SPS.constraint_set2_flag                     = uv( 1 );
	SPS.constraint_set3_flag                     = uv( 1 );
	SPS.constraint_set4_flag                     = uv( 1 );
	SPS.constraint_set5_flag                     = uv( 1 );
	uv( 2 ); // forbidden zero 2 bits
	SPS.level_idc                                = uv( 8 );
	SPS.seq_parameter_set_id                     = uev( );

	if( SPS.profile_idc == 100 || SPS.profile_idc == 110 || SPS.profile_idc == 122 || SPS.profile_idc == 244 || SPS.profile_idc == 44 || SPS.profile_idc == 83 || SPS.profile_idc == 86 || SPS.profile_idc == 118 || SPS.profile_idc == 128 || SPS.profile_idc == 138 || SPS.profile_idc == 139 || SPS.profile_idc == 134 )
	{
		SPS.chroma_format_idc                    = uev( );

		if( 3 == SPS.chroma_format_idc )
			SPS.separate_colour_plane_flag       = uv( 1 );

		SPS.bit_depth_luma_minus8                = uev( );
		SPS.bit_depth_chroma_minus8              = uev( );
		SPS.qpprime_y_zero_transform_bypass_flag = uv( 1 );
		SPS.seq_scaling_matrix_present_flag      = uv( 1 );

		if( SPS.seq_scaling_matrix_present_flag )
		{
			uint32_t for_value = ( SPS.chroma_format_idc != 3 ) ? 8 : 12;

			SPS.seq_scaling_list_present_flag = ( bool* )realloc( SPS.seq_scaling_list_present_flag, for_value * sizeof( bool ) );

			for( int i = 0; i < for_value; ++i )
				SPS.seq_scaling_list_present_flag[ i ] = uv( 1 );
		}
	}

	SPS.log2_max_frame_num_minus4                = uev( );
	SPS.pic_order_cnt_type                       = uev( );
	
	if( !SPS.pic_order_cnt_type )
		SPS.log2_max_pic_order_cnt_lsb_minus4    = uev( );

	else if( 1 == SPS.pic_order_cnt_type )
	{
		SPS.delta_pic_order_always_zero_flag     = uv( 1 );
		SPS.offset_for_non_ref_pic               = sev( );
		SPS.offset_for_top_to_bottom_field       = sev( );
		SPS.num_ref_frames_in_pic_order_cnt_cycle   = uev( );

		SPS.offset_for_ref_frame = ( int32_t* )realloc( SPS.offset_for_ref_frame, SPS.num_ref_frames_in_pic_order_cnt_cycle * sizeof( int32_t ) );

		for(int i = 0; i < SPS.num_ref_frames_in_pic_order_cnt_cycle; ++i)
		{
			SPS.offset_for_ref_frame[ i ]        = sev( );
		}
	}

	SPS.max_num_ref_frames                       = uev( );
	SPS.gaps_in_frame_num_value_allowed_flag     = uv( 1 );
	SPS.pic_width_in_mbs_minus1                  = uev( );
	SPS.pic_height_in_map_units_minus1           = uev( );
	SPS.frame_mbs_only_flag                      = uv( 1 );

	if( !SPS.frame_mbs_only_flag )
		SPS.mb_adaptive_frame_field_flag         = uv( 1 );

	SPS.direct_8x8_inference_flag                = uv( 1 );
	SPS.frame_cropping_flag                      = uv( 1 );

	if( SPS.frame_cropping_flag )
	{
		SPS.frame_crop_left_offset               = uev( );
		SPS.frame_crop_right_offset              = uev( );
		SPS.frame_crop_top_offset                = uev( );
		SPS.frame_crop_bottom_offset             = uev( );
	}

	SPS.vui_parameters_present_flag              = uv( 1 );
}

void H264parser::picPmSet( uint8_t nal_ref_idc, uint8_t nal_type )
{
	PPS.pic_parameter_set_id                     = uev( );
	PPS.seq_parameter_set_id                     = uev( );
	PPS.entropy_coding_mode_flag                 = uv( 1 );
	PPS.bottom_field_pic_order_in_frame_present_flag   = uv( 1 );
	PPS.num_slice_groups_minus1                  = uev( );

	if( PPS.num_slice_groups_minus1 )
	{
		PPS.slice_group_map_type                 = uev( );

		if( !PPS.slice_group_map_type )
		{
			PPS.run_length_minus1 = ( uint32_t* )realloc( PPS.run_length_minus1, PPS.num_slice_groups_minus1 * sizeof( uint32_t ) );

			for( int i = 0; i < PPS.num_slice_groups_minus1; ++i )
				PPS.run_length_minus1[ i ]       = uev( );
		}

		else if( 2 == PPS.slice_group_map_type )
		{
			PPS.top_left = ( uint32_t* )realloc( PPS.top_left, PPS.num_slice_groups_minus1 * sizeof( uint32_t ) );

			PPS.bottom_right = ( uint32_t* )realloc( PPS.bottom_right, PPS.num_slice_groups_minus1 * sizeof( uint32_t ) );

			for( int i = 0; i < PPS.num_slice_groups_minus1; ++i )
			{
				PPS.top_left[ i ]                = uev( );
				PPS.bottom_right[ i ]            = uev( );
			}
		}

		else if( 3 == PPS.slice_group_map_type || 4 == PPS.slice_group_map_type || 5 == PPS.slice_group_map_type )
		{
			PPS.slice_group_change_direction_flag   = uv( 1 );
			PPS.slice_group_change_rate_minus1   = uev( );
		}

		else if( 6 == PPS.slice_group_map_type )
		{
			PPS.pic_size_in_map_units_minus1     = uev( );

			PPS.slice_group_id = ( uint32_t* )realloc( PPS.slice_group_id, PPS.pic_size_in_map_units_minus1 * sizeof( uint32_t ) );

			uint32_t tempVal = ceil( log2( PPS.num_slice_groups_minus1 + 1) );

			for( int i = 0; i < PPS.pic_size_in_map_units_minus1; ++i )
				PPS.slice_group_id[ i ]              = uv( tempVal );
		}
	}

	PPS.num_ref_idx_l0_default_active_minus1     = uev( );
	PPS.num_ref_idx_l1_default_active_minus1     = uev( );
	PPS.weighted_pred_flag                       = uv( 1 );
	PPS.weighted_bipred_idc                      = uv( 2 );
	PPS.pic_init_qp_minus26                      = sev( );
	PPS.pic_init_qs_minus26                      = sev( );
	PPS.chroma_qp_index_offset                   = sev( );
	PPS.deblocking_filter_control_present_flag   = uv( 1 );
	PPS.constrained_intra_pred_flag              = uv( 1 );
	PPS.redundant_pic_cnt_present_flag           = uv( 1 );

	if( more_rbsp_data( ) )
	{
		std::cerr << "more rbsp data" << endl;

		PPS.transform_8x8_mode_flag              = uv( 1 );
		PPS.pic_scaling_matrix_present_flag      = uv( 1 );

		if( PPS.pic_scaling_matrix_present_flag )
		{
			uint32_t tempVal = 6 + ( ( SPS.chroma_format_idc != 3 ) ? 2 : 6) * PPS.transform_8x8_mode_flag;

			PPS.pic_scaling_list_present_flag = ( bool* )realloc( PPS.pic_scaling_list_present_flag, tempVal * sizeof( bool ) );

			for( int i = 0; i < tempVal; ++i )
				PPS.pic_scaling_list_present_flag[ i ]   = uv( 1 );
		}

		PPS.second_chroma_qp_index_offset        = sev( );

		std::cerr << "second chroma: " << PPS.second_chroma_qp_index_offset << endl;
	}
}

void H264parser::sliceHeader( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::refPicListMod( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::predWeightTable( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::decRefPicMark( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
};