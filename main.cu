#include <string>
#include <iostream>
#include <stdint.h>
#include <unistd.h>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

H264parser parser = H264parser( );

int frame_handler( uint8_t* start, uint32_t length )
{
	parser.parseFrame( start, length );
	return 0;
}

int main( int argc, char** argv )
{
	V4L2stream stream = V4L2stream( 1920, 1080, "/dev/video0", 8);
	stream.init( );
	stream.on( );
	for( int i = 0; i < 301; ++i)
		stream.getFrame( &frame_handler );
	stream.off( );

	return 0;
}