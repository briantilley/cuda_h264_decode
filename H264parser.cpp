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

	// proc params are used in a rolling array, probably unnecessary
	pPidx = -1;
	for( int i = 0; i < 6; ++i )
		procParams[ i ] = new CUVIDPROCPARAMS;

	// initialize decoded picture buffer as empty
	clearDPB( );
}

// set/get the BitPos member
BitPos H264parser::getPos( void )
{ return pos; }
void H264parser::setPos( BitPos in_pos )
{ pos = in_pos; }

// make a new object when the SH array needs to expand
slice_header* H264parser::makeSliceHeader( void )
{
	slice_header* retSH = new slice_header;

	retSH->pRPLM = new ref_pic_list_mod;
	retSH->pPWT  = new pred_weight_table;
	retSH->pDRPM = new dec_ref_pic_mark;

	return retSH;
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

// badly written with magic numbers
// not essential, will probably be removed
CUVIDPROCPARAMS* H264parser::getProcParams( void )
{
	if( 4 >= idx( ) && -1 < pPidx)
		return procParams[ ( pPidx + 2 ) % 6 ];
}

// get the current PicIdx
int32_t H264parser::idx( void )
{
	return cuvidPicParams->CurrPicIdx;
}

// H264 DATA VALUE PARSING FUNCTIONS-------------

// variable length unsigned int
uint32_t H264parser::uv  ( int numBits )
{
	return pos.readBits( numBits );
}

// unsigned exp-golomb int (process outlined in spec)
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

// signed exp-golomb int (process outlined in spec)
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

// END H264 DATA VALUE PARSING FUNCTIONS-------------

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

	// more bad magic numbers, needs to go
	for( int i = 0; i < 6; ++i )
		delete procParams[ i ];

	free( SH );

	free( cuvidPicParams );

	free( SDOs );
}