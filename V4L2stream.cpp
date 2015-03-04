// This is a wrapper class for V4L2
// meant to handle C style operation
// with a C++ interface.

#include <iostream>

#include "inc/constants.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

using std::max;
using std::min;

// simple member initialization
V4L2stream::V4L2stream( uint32_t in_width, uint32_t in_height, string in_device, uint32_t in_numBufs ): width( in_width ), height( in_height ), device( in_device ), numBufs( in_numBufs ) { }

// turn off the stream and close the device file as cleanup
V4L2stream::~V4L2stream( void )
{
	off( );
	close( fd );
}

// initialize the stream object, all V4L2 specific code
// some hardcoded magic values here
void V4L2stream::init( void )
{
	fd = open( device.c_str( ), O_RDWR );
	if( -1 == fd )
	{
		std::cerr << "failed to open " << device << endl;
		return;
	}

	if( -1 == xioctl( fd, VIDIOC_QUERYCAP, &device_caps) )
	{
		std::cerr << "error while querying caps" << endl;
		return;
	}

	format.type                  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	format.fmt.pix.width         = width;
	format.fmt.pix.height        = height;
	format.fmt.pix.pixelformat   = V4L2_PIX_FMT_H264;
	format.fmt.pix.field         = V4L2_FIELD_NONE;

	if( -1 == xioctl( fd, VIDIOC_S_FMT, &format ) )
	{
		std::cerr << "error while setting format" << endl;
		return;
	}

	ext_ctrls.count = 2;
	ext_ctrls.ctrl_class = V4L2_CTRL_CLASS_CAMERA;
	ext_ctrls.controls = ( v4l2_ext_control* )malloc( 2 * sizeof( v4l2_ext_control ) );

	ext_ctrls.controls[ 0 ].id     = V4L2_CID_EXPOSURE_AUTO;
	ext_ctrls.controls[ 0 ].value  = V4L2_EXPOSURE_MANUAL;
	ext_ctrls.controls[ 1 ].id     = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
	ext_ctrls.controls[ 1 ].value  = 0;

	if( -1 == xioctl( fd, VIDIOC_S_EXT_CTRLS, &ext_ctrls ) )
	{
		std::cerr << "error while setting controls" << endl;
		return;
	}

	request_bufs.count                 = numBufs;
	request_bufs.type                  = format.type;
	request_bufs.memory                = V4L2_MEMORY_MMAP;

	if( -1 == xioctl( fd, VIDIOC_REQBUFS, &request_bufs ) )
	{
		std::cerr << "error while requesting buffers" << endl;
		return;
	}

	buffer.type = request_bufs.type;
	buffer.memory = request_bufs.memory;

	buf_array = ( array_buf* )malloc( sizeof( array_buf ) * request_bufs.count );

	for(int i = 0; i < request_bufs.count; ++i)
	{
		buffer.index = i;
		if( -1 == xioctl( fd, VIDIOC_QUERYBUF, &buffer ) )
		{
			std::cerr << "error while querying buffer" << endl;
			return;
		}

		buf_array[ i ].start = ( uint8_t* )mmap( NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.m.offset );

		if( MAP_FAILED == buf_array[ i ].start )
		{
			std::cerr << "error mapping buf_array " << endl;
			return;
		}

		if( -1 == xioctl( fd, VIDIOC_QBUF, &buffer ) )
		{
			std::cerr << "error while initial enqueuing" << endl;
			return;
		}
	}
}

// turn the stream on
void V4L2stream::on( void )
{
	if( -1 == xioctl( fd, VIDIOC_STREAMON, &buffer.type ) )
	{
		std::cerr << "error while turning stream on" << endl;
		return;
	}

	buffer.index = 0;
}

// turn the stream off
void V4L2stream::off( void )
{
	if( -1 == xioctl( fd, VIDIOC_STREAMOFF, &buffer.type ) )
	{
		std::cerr << "error while turning stream off" << endl;
		return;
	}
}

int32_t V4L2stream::changeControl( V4L2stream_ctrl_type ctrl,
	int32_t ctrl_value,
	V4L2stream_change_type change )
{
	int32_t tempVal;

	if( V4L2stream_ABSOLUTE == change )
		switch( ctrl )
		{
			case V4L2stream_EXPOSURE:

				ext_ctrls.count = 1;
				ext_ctrls.ctrl_class = V4L2_CTRL_CLASS_CAMERA;
				ext_ctrls.controls = ( v4l2_ext_control* )malloc( sizeof( v4l2_ext_control ) );

				ext_ctrls.controls[ 0 ].id     = V4L2_CID_EXPOSURE_ABSOLUTE;
				ext_ctrls.controls[ 0 ].value  = max( 3, min( ctrl_value, 2047 ) );

				if( -1 == xioctl( fd, VIDIOC_S_EXT_CTRLS, &ext_ctrls ) )
				{
					perror( "V4L2stream::changeControl" );
					return 1;
				}

			break;
			case V4L2stream_NULL:
				std::cerr << "null control type is invalid" << endl;
				return 1;
			break;
			default:
				std::cerr << "control type not recognized" << endl;
				return 1;
			break;
		}

	else if( V4L2stream_RELATIVE == change )
		switch( ctrl )
		{
			case V4L2stream_EXPOSURE:

				ext_ctrls.count = 1;
				ext_ctrls.ctrl_class = V4L2_CTRL_CLASS_CAMERA;
				ext_ctrls.controls = ( v4l2_ext_control* )malloc( sizeof( v4l2_ext_control ) );

				ext_ctrls.controls[ 0 ].id     = V4L2_CID_EXPOSURE_ABSOLUTE;
				ext_ctrls.controls[ 0 ].size   = 0;
				
				if( -1 == xioctl( fd, VIDIOC_G_EXT_CTRLS, &ext_ctrls ) )
				{
					if( ENOSPC == errno )
					{
						cout << ext_ctrls.controls[ 0 ].size;
						if( -1 == xioctl( fd, VIDIOC_G_EXT_CTRLS, &ext_ctrls ) )
						{
							perror( "V4L2stream::changeControl ENOSPC == errno" );
							return 1;
						}
					}
					else
					{
						perror( "V4L2stream::changeControl ENOSPC != errno" );
						return 1;
					}
				}

				tempVal = ext_ctrls.controls[ 0 ].value + ctrl_value;
				
				tempVal = max( 3, min( tempVal, 2047 ) );
				
				ext_ctrls.controls[ 0 ].value = tempVal;
				
				if( -1 == xioctl( fd, VIDIOC_S_EXT_CTRLS, &ext_ctrls ) )
				{
					perror( "V4L2stream::changeControl" );
					return 1;
				}

			break;
			case V4L2stream_NULL:
				std::cerr << "null control type is invalid" << endl;
				return 1;
			break;
			default:
				std::cerr << "control type not recognized" << endl;
				return 1;
			break;
		}

	else
	{
		std::cerr << "change type not recognized" << endl;
		return 1;
	}

	return 0;
}

// retrieve one frame from V4L2, run the processing callback on the data,
// and give the frame buffer back to V4L2
void V4L2stream::getFrame( int32_t ( *ps_callback )( uint8_t*, uint32_t ) )
{
	
	if( -1 == xioctl( fd, VIDIOC_DQBUF, &buffer ) )
	{
		perror( "dequeuing buffer" );
		return;
	}

	if ( -1 == ps_callback( buf_array[ buffer.index ].start, buffer.bytesused ) )
		std::cerr << "frame processing callback failed" << endl;

	if( -1 == xioctl( fd, VIDIOC_QBUF, &buffer ) )
	{
		perror( "enqueuing buffer" );
		return;
	}

	++buffer.index; buffer.index %= request_bufs.count;

}

// ioctl wrapper function
int32_t V4L2stream::xioctl( int32_t file_desc, int32_t request, void* argp )
{
	int32_t retVal;

	do retVal = ioctl( file_desc, request, argp );
	while( -1 == retVal && EINTR == errno );

	return retVal;
}
