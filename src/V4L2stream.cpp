// This is a wrapper class for V4L2
// meant to handle C style operation
// with a C++ interface.

#include <iostream>

#include "inc/V4L2stream.h"

using std::cout;
using std::endl;
using std::string;

// simple member initialization
V4L2stream::V4L2stream( void ) { }

// turn off the stream and close the device file as cleanup
V4L2stream::~V4L2stream( void )
{
	off( );
	close( fd );
}

// initializing a device in V4L2 calls is simply a series of ioctl calls
// mostly pulled from online tutorials and V4L2 API specification
void V4L2stream::init( uint32_t* in_width, uint32_t* in_height, string in_device, uint32_t in_numBufs )
{
	width = *in_width;
	height = *in_height;
	device = in_device;
	numBufs = in_numBufs;

	// open device file
	fd = open( device.c_str( ), O_RDWR );
	if( -1 == fd )
	{
		std::cerr << "failed to open " << device << endl;
		exit( 1 );
	}

	// wuery capabilites (suggested by V4L2)
	if( -1 == xioctl( fd, VIDIOC_QUERYCAP, &device_caps) )
	{
		std::cerr << "error while querying caps" << endl;
		exit( 1 );
	}

	format.type                  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	format.fmt.pix.width         = width;
	format.fmt.pix.height        = height;
	format.fmt.pix.pixelformat   = PIXEL_FMT;
	format.fmt.pix.field         = V4L2_FIELD_NONE;

	// set video format
	if( -1 == xioctl( fd, VIDIOC_S_FMT, &format ) )
	{
		std::cerr << "error while setting format" << endl;
		exit( 1 );
	}

	// get and return the actual width and height
	width = format.fmt.pix.width;
	height = format.fmt.pix.height;
	*in_width = width;
	*in_height = height;

	ext_ctrls.count = 2;
	ext_ctrls.ctrl_class = V4L2_CTRL_CLASS_CAMERA;
	ext_ctrls.controls = ( v4l2_ext_control* )malloc( 2 * sizeof( v4l2_ext_control ) );

	ext_ctrls.controls[ 0 ].id     = V4L2_CID_EXPOSURE_AUTO;
	ext_ctrls.controls[ 0 ].value  = V4L2_EXPOSURE_MANUAL;
	ext_ctrls.controls[ 1 ].id     = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
	ext_ctrls.controls[ 1 ].value  = 0;

	// disable auto exposure (limits framerate)
	if( -1 == xioctl( fd, VIDIOC_S_EXT_CTRLS, &ext_ctrls ) )
	{
		std::cerr << "error while setting controls" << endl;
		exit( 1 );
	}

	request_bufs.count                 = numBufs;
	request_bufs.type                  = format.type;
	request_bufs.memory                = V4L2_MEMORY_MMAP;

	// request input buffers for webcam data
	if( -1 == xioctl( fd, VIDIOC_REQBUFS, &request_bufs ) )
	{
		std::cerr << "error while requesting buffers" << endl;
		exit( 1 );
	}

	// get the actual number of buffers
	numBufs = request_bufs.count;

	buffer.type = request_bufs.type;
	buffer.memory = request_bufs.memory;

	buf_array = ( array_buf* )malloc( sizeof( array_buf ) * numBufs );

	// make an array of buffers in V4L2 and enqueue them to prepare for stream on
	for(int i = 0; i < numBufs; ++i)
	{
		buffer.index = i;
		if( -1 == xioctl( fd, VIDIOC_QUERYBUF, &buffer ) )
		{
			std::cerr << "error while querying buffer" << endl;
			exit( 1 );
		}

		buf_array[ i ].start = ( uint8_t* )mmap( NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.m.offset );

		if( MAP_FAILED == buf_array[ i ].start )
		{
			std::cerr << "error mapping buf_array " << endl;
			exit( 1 );
		}

		if( -1 == xioctl( fd, VIDIOC_QBUF, &buffer ) )
		{
			std::cerr << "error while initial enqueuing" << endl;
			exit( 1 );
		}
	}
}

// turn the stream on
void V4L2stream::on( void )
{
	if( -1 == xioctl( fd, VIDIOC_STREAMON, &buffer.type ) )
	{
		std::cerr << "error while turning stream on" << endl;
		exit( 1 );
	}

	buffer.index = 0;
}

// turn the stream off
void V4L2stream::off( void )
{
	if( -1 == xioctl( fd, VIDIOC_STREAMOFF, &buffer.type ) )
	{
		std::cerr << "error while turning stream off" << endl;
		exit( 1 );
	}
}

// retrieve one frame from V4L2
// run the provided callback on the data,
// give the frame buffer back to V4L2
void V4L2stream::getCodedFrame( int ( *ps_callback )( uint8_t*, uint32_t ) )
{
	
	if( -1 == xioctl( fd, VIDIOC_DQBUF, &buffer ) )
	{
		std::cerr << "error while retrieving frame" << endl;
		exit( 1 );
	}

	if ( -1 == ps_callback( buf_array[ buffer.index ].start, buffer.bytesused ) )
		cout << "frame processing callback failed" << endl;

	if( -1 == xioctl( fd, VIDIOC_QBUF, &buffer ) )
	{
		std::cerr << "error while releasing buffer" << endl;
		exit( 1 );
	}

	++buffer.index; buffer.index %= numBufs;

}