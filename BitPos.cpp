#include <string>
#include <iostream>
#include <stdint.h>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// initialize to NULL byte with MSB
BitPos::BitPos(): byte( NULL ), mask( 0x80 ) { }

// initialize to MSB of given byte
BitPos::BitPos( uint8_t* in_byte ): byte( in_byte ), mask( 0x80 ) { }

// initialize with given position info
BitPos::BitPos( uint8_t* in_byte, uint8_t in_mask ): byte( in_byte ), mask( in_mask ) { }

uint8_t* BitPos::getByte( void )
{ return byte; }
void BitPos::setByte( uint8_t* in_byte )
{ byte = in_byte; }

uint8_t BitPos::getMask( void )
{ return mask; }
void BitPos::setMask( uint8_t in_mask )
{ mask = in_mask; }

void BitPos::advance( void )
{
	mask >>= 1;
	if(!mask) { ++byte; mask = 0x80; }
}

uint32_t BitPos::readBits( int numBits )
{
	uint32_t retVal = 0;

	retVal |= ( *byte & mask ) ? 1 : 0;
	advance( );

	for( int i = 0; i < numBits - 1; ++i )
	{
		retVal <<= 1;

		retVal += ( *byte & mask ) ? 1 : 0;
		advance( );
	}

	return retVal;
}

BitPos::~BitPos( ) { }