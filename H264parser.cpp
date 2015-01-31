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

void H264parser::update( nal_type type ) { update( pos, type ); }
void H264parser::update( BitPos in_pos, nal_type type )
{
	pos = in_pos;
	// ...
}

void H264parser::populate( CUVIDPICPARAMS &picPms )
{
	// ...
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