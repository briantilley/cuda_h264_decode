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

H264parser::H264parser( void ): pos( BitPos( ) )
{
	SH.pRPLM = new ref_pic_list_mod;
	SH.pPWT  = new pred_weight_table;
	SH.pDRPM = new dec_ref_pic_mark;
}

H264parser::H264parser( BitPos in_pos ): pos( in_pos )
{
	SH.pRPLM = new ref_pic_list_mod;
	SH.pPWT  = new pred_weight_table;
	SH.pDRPM = new dec_ref_pic_mark;
}

BitPos H264parser::getPos( void )
{ return pos; }
void H264parser::setPos( BitPos in_pos )
{ pos = in_pos; }

void H264parser::parseFrame( uint32_t length ) { parseFrame( pos.getByte( ), length ); }
void H264parser::parseFrame( const uint8_t* start, uint32_t length )
{
	pos.setByte( ( uint8_t* )start );

	uint8_t nal_ref_idc;
	uint8_t nal_type;
	uint32_t comp_buf;

	while( true )
	{
		pos.setMask( 0x80 );

		nal_ref_idc = 0;
		nal_type    = 0;
		comp_buf    = 0;

		while( 1 != comp_buf )
		{
			comp_buf <<= 8;
			comp_buf  += pos.readByte( );

			if( pos.getByte( ) >= ( start + length ) ) return; // end of frame
		}

		uv( 1 ); // forbidden zero bit
		
		nal_ref_idc = uv( 2 );
		nal_type    = uv( 5 );

		switch( nal_type )
		{
			case 0x01:

				// std::cerr << ".";
				sliceHeader( nal_ref_idc, nal_type );
				break;

			case 0x05:

				// std::cerr << ".";
				sliceHeader( nal_ref_idc, nal_type );
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

bool H264parser::more_rbsp_data( void )
{
	BitPos pos_copy = BitPos( pos );
	uint32_t comp_buf = 0;
	uint8_t diff      = 0;

	while( 1 != comp_buf )
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

H264parser::~H264parser( void )
{
	delete SH.pRPLM;
	delete SH.pPWT;
	delete SH.pDRPM;
}