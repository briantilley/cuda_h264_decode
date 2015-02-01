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

H264parser::H264parser( void ): pos( BitPos( ) ) {}

H264parser::H264parser( BitPos in_pos ): pos( in_pos ) { }

BitPos H264parser::getPos( void )
{ return pos; }
void H264parser::setPos( BitPos in_pos )
{ pos = in_pos; }

void H264parser::parseFrame( void ) { parseFrame( pos ); }
void H264parser::parseFrame( BitPos in_pos )
{
	pos = in_pos;

	uint8_t nal_ref_idc = 0;
	uint8_t nal_type    = 0;
	uint32_t comp_buf   = 0;

	while( 1 != comp_buf )
	{
		comp_buf <<= 8;
		comp_buf  += pos.readBits( 8 );
	}

	uv( 1 ); // forbidden zero bit
	
	nal_ref_idc = uv( 2 );
	nal_type    = uv( 5 );

	switch( nal_type )
	{
		case 0x01:

			sliceHeader( nal_ref_idc, nal_type );

		break;
		case 0x05:

			sliceHeader( nal_ref_idc, nal_type );

		break;
		case 0x07:

			seqPmSet( nal_ref_idc, nal_type );

		break;
		case 0x08:

			picPmSet( nal_ref_idc, nal_type );

		break;
	}

}

bool H264parser::flag( void )
{
	return pos.readBits( 1 ) ? true : false;
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

H264parser::~H264parser( void ) { }