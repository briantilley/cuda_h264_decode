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

int main( int argc, char** argv )
{
	uint8_t test[ 4 ] = { 0x00, 0xb5, 0x4e, 0x32 };
	H264parser parser = H264parser( );

	parser.setPos( BitPos( ( uint8_t* )&test[ 0 ] ) );

	cout << "sev( ): " << parser.sev( ) << endl;

	return 0;
}