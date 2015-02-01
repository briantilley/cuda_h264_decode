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

int main( int argc, char** argv )
{
	V4L2stream stream = V4L2stream( );
	stream.init( );
	stream.on( );
	sleep( 5 );
	stream.off( );

	return 0;
}