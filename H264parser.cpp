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

// overloaded constructor that uses defaults or inital values
H264parser::H264parser( void ): pos( BitPos( ) ), maxSHcount( DEFAULT_SH_COUNT )
{ init( ); }

H264parser::H264parser( BitPos in_pos ): pos( in_pos ), maxSHcount( DEFAULT_SH_COUNT )
{ init( ); }

// constructor initialization routine to stay clean
inline void H264parser::init( void )
{
	// each frame has multiple SH units,
	// so pointers to each are stored in an array
	SH = ( slice_header** )malloc( maxSHcount * sizeof( slice_header* ) ) ;
	for( int i = 0; i < maxSHcount; ++i )
		SH[ i ] = makeSliceHeader( ); // simple allocation routine

	PrevFrameNum = -1; // used to detect if second or first field in a pair

	// parser object has its own pic params struct
	cuvidPicParams = ( CUVIDPICPARAMS* )malloc( sizeof( CUVIDPICPARAMS ) );
	cuvidPicParams->CurrPicIdx = -1; // PicIdx is incremented before use

	// slice data offsets, one for each slice header
	SDOs = ( uint32_t* )malloc( DEFAULT_SH_COUNT * sizeof( uint32_t ) );

	// initialize decoded picture buffer as empty
	clearDPB( );
}

// destructor only frees dynamic memory allocations
H264parser::~H264parser( void )
{
	for( int i = 0; i < maxSHcount; ++i )
	{
		delete SH[ i ]->pRPLM;
		delete SH[ i ]->pPWT;
		delete SH[ i ]->pDRPM;

		delete SH[ i ];
	}

	free( SH );

	free( cuvidPicParams );

	free( SDOs );
}

// make a new object when the SH array needs to expand
slice_header* H264parser::makeSliceHeader( void )
{
	slice_header* retSH = new slice_header;

	retSH->pRPLM = new ref_pic_list_mod;
	retSH->pPWT  = new pred_weight_table;
	retSH->pDRPM = new dec_ref_pic_mark;

	return retSH;
}

// set/get the BitPos member
BitPos H264parser::getPos( void )
{ return pos; }
void H264parser::setPos( BitPos in_pos )
{ pos = in_pos; }

// get the current PicIdx
int32_t H264parser::idx( void )
{
	return cuvidPicParams->CurrPicIdx;
}

// overloaded frame parsing (must pass length, current BitPos used by default)
void H264parser::parseFrame( uint32_t in_length ) { parseFrame( pos.getByte( ), in_length ); }
void H264parser::parseFrame( const uint8_t* in_start, uint32_t in_length )
{
	// basic stream data
	start = in_start;
	length = in_length;

	// set the starting point (bits/byte are parts in a state-based nature)
	pos.setByte( ( uint8_t* )start );

	uint8_t nal_type; // integer reference to the type of NAL unit found
	uint32_t comp_buf; // stores previous bytes of the stream to find NAL units

	SHidx = 0; // start at the first slice header
	idr_pic_flag = true; // pic is assumed intra-coded until non-IDR slice found

    ++cuvidPicParams->CurrPicIdx; // increment the index before parsing (-1 initially)

    // continuously loop to find all NAL units
    // nested while loop for seeking returns at end of frame
	while( true )
	{
		pos.setMask( 0x80 ); // start at the MSB

		nal_type    = 0; // clear previous type
		comp_buf    = 0; // clear seek buffer

		while( NAL_UNIT_START != comp_buf )
		{
			// shift a byte for comparison
			comp_buf <<= 8;
			comp_buf  += pos.readByte( );

			// return when end is reached
			if( pos.getByte( ) >= ( start + length ) )
			{
				fillParams( );
				return; // end of frame
			}
		}

		// initial NAL parsing begins here
		uv( 1 ); // forbidden zero bit in H264 spec
		
		nal_ref_idc = uv( 2 ); // 0 if not a reference frame
		nal_type    = uv( 5 );

		if( 1 == nal_type ) idr_pic_flag = false; // uses inter-coding

		// follow proper parsing routine based on NAL unit type
		switch( nal_type )
		{
			case 0x01: case 0x05: // slice header, nonIDR and IDR resp.

				// expand SH array if necessary
				if( SHidx >= maxSHcount )
				{
					SH = ( slice_header** )realloc( SH, ( SHidx + 1 ) * sizeof( slice_header* ) );
					SH[ SHidx ] = makeSliceHeader( );

					SDOs = ( uint32_t* )realloc( SDOs, maxSHcount * sizeof( uint32_t ) );
				}

				// parse data based on H264 spec
				sliceHeader( nal_ref_idc, nal_type );

				// manage frame nums using the first SH struct
				if( !SHidx )
				{
					cuvidPicParams->second_field = ( PrevFrameNum == SH[ 0 ]->frame_num && SH[ 0 ]->field_pic_flag ) ? 1 : 0;
					PrevFrameNum = SH[ 0 ]->frame_num;
				}

				++SHidx; // SHidx = physical count of SH units after parsing
				maxSHcount = std::max( SHidx, maxSHcount ); // track size of SH array

				break;

			case 0x07: // sequence parameter set

				seqPmSet( nal_ref_idc, nal_type ); // based on H264 spec
				break;

			case 0x08: // picture parameter set

				picPmSet( nal_ref_idc, nal_type ); // based on H264 spec
				break;
		}
	}
}

// H264 DATA VALUE PARSING FUNCTIONS-------------

// variable length unsigned int
uint32_t H264parser::uv  ( int numBits )
{
	return pos.readBits( numBits );
}

// unsigned exp-golomb int (process outlined in spec)
uint32_t H264parser::uev ( void )
{
	uint8_t  numZeroes = 0;
	uint32_t code      = 0;
	bool     oneFound  = false;

	while( !( oneFound ) )
	{
		if( pos.readBits( 1 ) )
			oneFound = true;
		else
			++numZeroes;
	}

	code = 1 << numZeroes;
	code += pos.readBits( numZeroes ) - 1;

	return code;
}

// signed exp-golomb int (process outlined in spec)
int32_t  H264parser::sev ( void )
{
	uint32_t numZeroes = 0;
	uint32_t code      = 0;
	bool     oneFound  = false;
	int32_t  value     = 0;

	while( !( oneFound ) )
	{
		if( pos.readBits( 1 ) )
			oneFound = true;
		else
			++numZeroes;
	}

	code = 1 << numZeroes;
	code += pos.readBits( numZeroes ) - 1;

	value = ( code + 1 ) / 2; // depends on taking the floor value of division
	
	if( !( code % 2 ) )
		value *= -1;

	return value;
}

// END H264 DATA VALUE PARSING FUNCTIONS-------------

// NAL unit specific methods next 497 lines
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
			{
				SPS.seq_scaling_list_present_flag[ i ] = uv( 1 );

				if( SPS.seq_scaling_list_present_flag[ i ] )
				{
					if( i < 6 )
						scaling_list( cuvidPicParams->CodecSpecific.h264.WeightScale4x4[ i ], 16, &defaultMatrix4x4[ i ] );
					else
						scaling_list( cuvidPicParams->CodecSpecific.h264.WeightScale8x8[ i - 6], 64, &defaultMatrix8x8[ i - 6 ] );
				}
			}
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
		PPS.transform_8x8_mode_flag              = uv( 1 );
		PPS.pic_scaling_matrix_present_flag      = uv( 1 );

		if( PPS.pic_scaling_matrix_present_flag )
		{
			uint32_t tempVal = 6 + ( ( SPS.chroma_format_idc != 3 ) ? 2 : 6) * PPS.transform_8x8_mode_flag;

			PPS.pic_scaling_list_present_flag = ( bool* )realloc( PPS.pic_scaling_list_present_flag, tempVal * sizeof( bool ) );

			for( int i = 0; i < tempVal; ++i )
			{
				PPS.pic_scaling_list_present_flag[ i ]   = uv( 1 );

				if( PPS.pic_scaling_list_present_flag[ i ] )
				{
					if( i < 6 )
						scaling_list( cuvidPicParams->CodecSpecific.h264.WeightScale4x4[ i ], 16, &defaultMatrix4x4[ i ] );
					else
						scaling_list( cuvidPicParams->CodecSpecific.h264.WeightScale8x8[ i - 6], 64, &defaultMatrix8x8[ i - 6 ] );
				}
			}
		}

		PPS.second_chroma_qp_index_offset        = sev( );
	}
}

void H264parser::sliceHeader( uint8_t nal_ref_idc, uint8_t nal_type )
{
	SH[ SHidx ]->first_mb_in_slice                         = uev( );
	SH[ SHidx ]->slice_type                                = uev( );
	SH[ SHidx ]->pic_parameter_set_id                      = uev( );

	if( SPS.separate_colour_plane_flag )
		SH[ SHidx ]->colour_plane_id                       = uv( 2 );

	SH[ SHidx ]->frame_num                                 = uv( SPS.log2_max_frame_num_minus4 + 4 );

	if( !SPS.frame_mbs_only_flag )
	{
		SH[ SHidx ]->field_pic_flag                        = uv( 1 );

		if( SH[ SHidx ]->field_pic_flag )
			SH[ SHidx ]->bottom_field_flag                 = uv( 1 );
	}

	if( 5 == nal_type )
		SH[ SHidx ]->idr_pic_id                            = uev( );

	if( !SPS.pic_order_cnt_type )
	{
		SH[ SHidx ]->pic_order_cnt_lsb                     = uv( SPS.log2_max_pic_order_cnt_lsb_minus4 + 4 );

		if( PPS.bottom_field_pic_order_in_frame_present_flag && !SH[ SHidx ]->field_pic_flag )
			SH[ SHidx ]->delta_pic_order_cnt_bottom        = sev( );
	}

	else if( 1 == SPS.pic_order_cnt_type && !SPS.delta_pic_order_always_zero_flag )
	{
		SH[ SHidx ]->delta_pic_order_cnt[ 0 ]              = sev( );

		if( PPS.bottom_field_pic_order_in_frame_present_flag && !SH[ SHidx ]->field_pic_flag )
			SH[ SHidx ]->delta_pic_order_cnt[ 1 ]          = sev( );
	}

	if( PPS.redundant_pic_cnt_present_flag )
		SH[ SHidx ]->redundant_pic_cnt                     = uev( );

	if( 1 == SH[ SHidx ]->slice_type % 5 )
		SH[ SHidx ]->direct_spatial_mv_pred_flag           = uv( 1 );

	if( 0 == SH[ SHidx ]->slice_type % 5 || 1 == SH[ SHidx ]->slice_type % 5 || 3 == SH[ SHidx ]->slice_type % 5 )
	{
		SH[ SHidx ]->num_ref_idx_active_override_flag      = uv( 1 );

		if( SH[ SHidx ]->num_ref_idx_active_override_flag )
		{
			SH[ SHidx ]->num_ref_idx_l0_active_minus1      = uev( );

			if( 1 == SH[ SHidx ]->slice_type % 5 )
				SH[ SHidx ]->num_ref_idx_l1_active_minus1  = uev( );
		}
	}

	if( 20 != nal_type && 21 != nal_type )
		refPicListMod( nal_ref_idc, nal_type );

	if( ( PPS.weighted_pred_flag && ( 0 == SH[ SHidx ]->slice_type % 5 || 3 == SH[ SHidx ]->slice_type % 5 ) ) || ( 1 == PPS.weighted_bipred_idc && 1 == SH[ SHidx ]->slice_type % 5 ) )
		predWeightTable( nal_ref_idc, nal_type );

	if( nal_ref_idc )
		decRefPicMark( nal_ref_idc, nal_type );

	if( PPS.entropy_coding_mode_flag && 2 != SH[ SHidx ]->slice_type % 5 && 4 != SH[ SHidx ]->slice_type % 5)
		SH[ SHidx ]->cabac_init_idc                        = uev( );

	SH[ SHidx ]->slice_qp_delta                            = sev( );

	if( 3 == SH[ SHidx ]->slice_type % 5 || 4 == SH[ SHidx ]->slice_type % 5 )
	{
		if( 3 == SH[ SHidx ]->slice_type % 5 )
			SH[ SHidx ]->sp_for_switch_flag                = uv( 1 );

		SH[ SHidx ]->slice_qs_delta                        = sev( );
	}

	if( PPS.deblocking_filter_control_present_flag )
	{
		SH[ SHidx ]->disable_deblocking_filter_idc           = uev( );

		if( 1 != SH[ SHidx ]->disable_deblocking_filter_idc )
		{
			SH[ SHidx ]->slice_alpha_c0_offset_div2        = sev( );
			SH[ SHidx ]->slice_beta_offset_div2            = sev( );
		}
	}

	if( PPS.num_slice_groups_minus1 && PPS.slice_group_map_type >= 3 && PPS.slice_group_map_type <= 5 )
	{
		uint32_t PicSizeInMapUnits = ( SPS.pic_width_in_mbs_minus1 + 1 ) * ( SPS.pic_height_in_map_units_minus1 + 1 );
		uint32_t SliceGroupChangeRate = PPS.slice_group_change_rate_minus1 + 1;
		uint32_t tempVal = ceil( log2( PicSizeInMapUnits / SliceGroupChangeRate + 1 ) );

		SH[ SHidx ]->slice_group_change_cycle              = uv( tempVal );
	}

	SDOs[ SHidx ] = pos.getByte( ) - start;
}

void H264parser::refPicListMod( uint8_t nal_ref_idc, uint8_t nal_type )
{
	if( 2 != SH[ SHidx ]->slice_type % 5 && 4 != SH[ SHidx ]->slice_type % 5 )
	{
		SH[ SHidx ]->pRPLM->ref_pic_list_modification_flag_l0   = uv( 1 );

		if( SH[ SHidx ]->pRPLM->ref_pic_list_modification_flag_l0 )
		{
			do
			{
				SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc   = uev( );

				if( 1 >= SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc )
					SH[ SHidx ]->pRPLM->abs_diff_pic_num_minus1   = uev( );
				else if( 2 == SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc )
					SH[ SHidx ]->pRPLM->long_term_pic_num         = uev( );
			
			}while( 3 != SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc );
		}
	}

	if( 1 == SH[ SHidx ]->slice_type % 5 )
	{
		SH[ SHidx ]->pRPLM->ref_pic_list_modification_flag_l1   = uv( 1 );

		if( SH[ SHidx ]->pRPLM->ref_pic_list_modification_flag_l0 )
		{
			do
			{
				SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc   = uev( );

				if( 1 >= SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc )
					SH[ SHidx ]->pRPLM->abs_diff_pic_num_minus1   = uev( );
				else if( 2 == SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc )
					SH[ SHidx ]->pRPLM->long_term_pic_num         = uev( );
			
			}while( 3 != SH[ SHidx ]->pRPLM->modification_of_pic_nums_idc );
		}
	}
}

void H264parser::predWeightTable( uint8_t nal_ref_idc, uint8_t nal_type )
{

	SH[ SHidx ]->pPWT->luma_log2_weight_denom              = uev( );

	uint32_t ChromaArrayType = ( !SPS.separate_colour_plane_flag ) ? SPS.chroma_format_idc : 0;
	
	if( ChromaArrayType )
		SH[ SHidx ]->pPWT->chroma_log2_weight_denom        = uev( );

	SH[ SHidx ]->pPWT->luma_weight_l0 = ( int32_t* )realloc( SH[ SHidx ]->pPWT->luma_weight_l0, SH[ SHidx ]->num_ref_idx_l0_active_minus1 * sizeof( int32_t ) );
	SH[ SHidx ]->pPWT->luma_offset_l0 = ( int32_t* )realloc( SH[ SHidx ]->pPWT->luma_offset_l0, SH[ SHidx ]->num_ref_idx_l0_active_minus1 * sizeof( int32_t ) );

	SH[ SHidx ]->pPWT->chroma_weight_l0 = ( int32_t** )realloc( SH[ SHidx ]->pPWT->chroma_weight_l0, SH[ SHidx ]->num_ref_idx_l0_active_minus1 * sizeof( int32_t* ) );
	SH[ SHidx ]->pPWT->chroma_offset_l0 = ( int32_t** )realloc( SH[ SHidx ]->pPWT->chroma_offset_l0, SH[ SHidx ]->num_ref_idx_l0_active_minus1 * sizeof( int32_t* ) );

	for ( int i = 0; i < SH[ SHidx ]->num_ref_idx_l0_active_minus1; ++i )
	{
		SH[ SHidx ]->pPWT->luma_weight_l0_flag             = uv( 1 );

		if( SH[ SHidx ]->pPWT->luma_weight_l0_flag )
		{
			SH[ SHidx ]->pPWT->luma_weight_l0[ i ]         = sev( );
			SH[ SHidx ]->pPWT->luma_offset_l0[ i ]         = sev( );
		}

		if( ChromaArrayType )
		{
			SH[ SHidx ]->pPWT->chroma_weight_l0_flag       = uv( 1 );

			if( SH[ SHidx ]->pPWT->chroma_weight_l0_flag )
			{
				SH[ SHidx ]->pPWT->chroma_weight_l0[ i ] = ( int32_t* )realloc( SH[ SHidx ]->pPWT->chroma_weight_l0[ i ], 2 * sizeof( int32_t ) );

				SH[ SHidx ]->pPWT->chroma_offset_l0[ i ] = ( int32_t* )realloc( SH[ SHidx ]->pPWT->chroma_offset_l0[ i ], 2 * sizeof( int32_t ) );

				for( int j = 0; j < 2; ++j )
				{
					SH[ SHidx ]->pPWT->chroma_weight_l0[ i ][ j ]   = sev( );
					SH[ SHidx ]->pPWT->chroma_offset_l0[ i ][ j ]   = sev( );
				}
			}
		}
	}

	if( 1 == SH[ SHidx ]->slice_type % 5 )
	{
		SH[ SHidx ]->pPWT->luma_weight_l1 = ( int32_t* )realloc( SH[ SHidx ]->pPWT->luma_weight_l1, SH[ SHidx ]->num_ref_idx_l1_active_minus1 * sizeof( int32_t ) );
		SH[ SHidx ]->pPWT->luma_offset_l1 = ( int32_t* )realloc( SH[ SHidx ]->pPWT->luma_offset_l1, SH[ SHidx ]->num_ref_idx_l1_active_minus1 * sizeof( int32_t ) );

		SH[ SHidx ]->pPWT->chroma_weight_l1 = ( int32_t** )realloc( SH[ SHidx ]->pPWT->chroma_weight_l1, SH[ SHidx ]->num_ref_idx_l1_active_minus1 * sizeof( int32_t* ) );
		SH[ SHidx ]->pPWT->chroma_offset_l1 = ( int32_t** )realloc( SH[ SHidx ]->pPWT->chroma_offset_l1, SH[ SHidx ]->num_ref_idx_l1_active_minus1 * sizeof( int32_t* ) );

		for ( int i = 0; i < SH[ SHidx ]->num_ref_idx_l1_active_minus1; ++i )
		{
			SH[ SHidx ]->pPWT->luma_weight_l1_flag             = uv( 1 );

			if( SH[ SHidx ]->pPWT->luma_weight_l1_flag )
			{
				SH[ SHidx ]->pPWT->luma_weight_l1[ i ]         = sev( );
				SH[ SHidx ]->pPWT->luma_offset_l1[ i ]         = sev( );
			}

			if( ChromaArrayType )
			{
				SH[ SHidx ]->pPWT->chroma_weight_l1_flag       = uv( 1 );

				if( SH[ SHidx ]->pPWT->chroma_weight_l1_flag )
				{
					SH[ SHidx ]->pPWT->chroma_weight_l1[ i ] = ( int32_t* )realloc( SH[ SHidx ]->pPWT->chroma_weight_l1[ i ], 2 * sizeof( int32_t ) );

					SH[ SHidx ]->pPWT->chroma_offset_l1[ i ] = ( int32_t* )realloc( SH[ SHidx ]->pPWT->chroma_offset_l1[ i ], 2 * sizeof( int32_t ) );

					for( int j = 0; j < 2; ++j )
					{
						SH[ SHidx ]->pPWT->chroma_weight_l1[ i ][ j ]   = sev( );
						SH[ SHidx ]->pPWT->chroma_offset_l1[ i ][ j ]   = sev( );
					}
				}
			}
		}
	}
}

void H264parser::decRefPicMark( uint8_t nal_ref_idc, uint8_t nal_type )
{
	if( 5 == nal_type )
	{
		SH[ SHidx ]->pDRPM->no_output_of_prior_pics_flag   = uv( 1 );
		SH[ SHidx ]->pDRPM->long_term_reference_flag       = uv( 1 );
	}

	else
	{
		SH[ SHidx ]->pDRPM->adaptive_ref_pic_marking_mode_flag   = uv( 1 );

		if( SH[ SHidx ]->pDRPM->adaptive_ref_pic_marking_mode_flag )
		{
			do
			{
				SH[ SHidx ]->pDRPM->memory_management_control_operation   = uev( );

				if( 1 == SH[ SHidx ]->pDRPM->memory_management_control_operation || 3 == SH[ SHidx ]->pDRPM->memory_management_control_operation)
					SH[ SHidx ]->pDRPM->difference_of_pic_nums_minus1   = uev( );

				if( 2 == SH[ SHidx ]->pDRPM->memory_management_control_operation)
					SH[ SHidx ]->pDRPM->long_term_pic_num  = uev( );

				if( 3 == SH[ SHidx ]->pDRPM->memory_management_control_operation || 6 == SH[ SHidx ]->pDRPM->memory_management_control_operation)
					SH[ SHidx ]->pDRPM->long_term_frame_idx   = uev( );

				if( 4 == SH[ SHidx ]->pDRPM->memory_management_control_operation)
					SH[ SHidx ]->pDRPM->max_long_term_frame_idx_plus1   = uev( );

			} while( SH[ SHidx ]->pDRPM->memory_management_control_operation );
		}
	}
}

void H264parser::scaling_list( uint8_t* scalingList, uint8_t listSize, bool* defaultMatrix)
{
	uint8_t lastScale   = 8;
	uint8_t nextScale   = 8;
	int32_t delta_scale = 0;

	for( int i = 0; i < listSize; ++i )
	{
		if( nextScale )
		{
			delta_scale    = sev( );
			nextScale      = ( lastScale + delta_scale ) % 256;

			*defaultMatrix = ( !i && !nextScale );
		}

		scalingList[ i ]   = ( !nextScale ) ? lastScale : nextScale;
		lastScale          = scalingList[ i ];
	}
}

// check if there is more data in the RBSP
// not sure how exact or robust this method is
bool H264parser::more_rbsp_data( void )
{
	BitPos pos_copy = BitPos( pos );
	uint32_t comp_buf = 0;
	uint8_t diff      = 0;

	while( NAL_UNIT_START != comp_buf )
	{
		comp_buf <<= 8;
		comp_buf  += pos.readByte( );
	}

	pos.setByte( pos.getByte( ) - 4 );
	pos.setMask( 0x01 );

	while( 2 != comp_buf )
	{
		comp_buf >>= 1;
		comp_buf  |= pos.readBitReverse( ) << 1;
		comp_buf  &= 0x03;
	}

	diff = pos.getByte( ) - pos_copy.getByte( );
	pos = BitPos( pos_copy );

	if( diff >= 2 )
		return true;
	else
		return false;
}

// cuvid handling methods next 102 lines

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