#include <string>
#include <iostream>
#include <nvcuvid.h>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

H264parser::H264parser( void ): pos( BitPos( ) ), maxSHcount( DEFAULT_SH_COUNT )
{ init( ); }

H264parser::H264parser( BitPos in_pos ): pos( in_pos ), maxSHcount( DEFAULT_SH_COUNT )
{ init( ); }

inline void H264parser::init( void )
{
	SH = ( slice_header** )malloc( maxSHcount * sizeof( slice_header* ) ) ;
	for( int i = 0; i < maxSHcount; ++i )
		SH[ i ] = makeSliceHeader( );

	PrevFrameNum = -1;

	cuvidPicParams = ( CUVIDPICPARAMS* )malloc( sizeof( CUVIDPICPARAMS ) );
	cuvidPicParams->CurrPicIdx = -1;

	SDOs = ( uint32_t* )malloc( DEFAULT_SH_COUNT * sizeof( uint32_t ) );

	pPidx = -1;
	for( int i = 0; i < 6; ++i )
		procParams[ i ] = new CUVIDPROCPARAMS;

	// initialize DPB as empty
	clearDPB( );
}

BitPos H264parser::getPos( void )
{ return pos; }
void H264parser::setPos( BitPos in_pos )
{ pos = in_pos; }

slice_header* H264parser::makeSliceHeader( void )
{
	slice_header* retSH = new slice_header;

	retSH->pRPLM = new ref_pic_list_mod;
	retSH->pPWT  = new pred_weight_table;
	retSH->pDRPM = new dec_ref_pic_mark;

	return retSH;
}

void H264parser::parseFrame( uint32_t in_length ) { parseFrame( pos.getByte( ), in_length ); }
void H264parser::parseFrame( const uint8_t* in_start, uint32_t in_length )
{
	// cout << ( uint16_t )maxSHcount << " " << std::flush;
	start = in_start;
	length = in_length;

	pos.setByte( ( uint8_t* )start );

	uint8_t nal_type;
	uint32_t comp_buf;

	SHidx = 0;
	idr_pic_flag = true;

    ++cuvidPicParams->CurrPicIdx;

	while( true )
	{
		pos.setMask( 0x80 );

		nal_type    = 0;
		comp_buf    = 0;

		while( NAL_UNIT_START != comp_buf )
		{
			comp_buf <<= 8;
			comp_buf  += pos.readByte( );

			if( pos.getByte( ) >= ( start + length ) )
			{
				fillParams( );
				return; // end of frame
			}
		}

		uv( 1 ); // forbidden zero bit
		
		nal_ref_idc = uv( 2 );
		nal_type    = uv( 5 );

		if( 1 == nal_type ) idr_pic_flag = false;

		switch( nal_type )
		{
			case 0x01: case 0x05:

				if( SHidx >= maxSHcount )
				{
					SH = ( slice_header** )realloc( SH, ( SHidx + 1 ) * sizeof( slice_header* ) );
					SH[ SHidx ] = makeSliceHeader( );

					SDOs = ( uint32_t* )realloc( SDOs, maxSHcount * sizeof( uint32_t ) );
				}

				sliceHeader( nal_ref_idc, nal_type );

				if( !SHidx )
				{
					cuvidPicParams->second_field = ( PrevFrameNum == SH[ 0 ]->frame_num && SH[ 0 ]->field_pic_flag ) ? 1 : 0;
					PrevFrameNum = SH[ 0 ]->frame_num;
				}

				++SHidx; // SHidx = total count of frame's SH units after parsing
				maxSHcount = std::max( SHidx, maxSHcount );

				break;

			case 0x07:

				// std::cerr << "SPS" << endl;
				seqPmSet( nal_ref_idc, nal_type );
				break;

			case 0x08:

				// std::cerr << "PPS" << endl;
				picPmSet( nal_ref_idc, nal_type );
				break;
		}
	}
}

CUVIDPROCPARAMS* H264parser::getProcParams( void )
{
	if( 4 >= idx( ) && -1 < pPidx)
		return procParams[ ( pPidx + 2 ) % 6 ];
}

int32_t H264parser::idx( void )
{
	return cuvidPicParams->CurrPicIdx;
}

uint32_t H264parser::uv  ( int numBits )
{
	return pos.readBits( numBits );
}

uint32_t H264parser::uev ( void ) // not working
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

int32_t  H264parser::sev ( void ) // not working
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

H264parser::~H264parser( void )
{
	for( int i = 0; i < maxSHcount; ++i )
	{
		delete SH[ i ]->pRPLM;
		delete SH[ i ]->pPWT;
		delete SH[ i ]->pDRPM;

		delete SH[ i ];
	}

	for( int i = 0; i < 6; ++i )
		delete procParams[ i ];

	free( SH );

	free( cuvidPicParams );

	free( SDOs );
}