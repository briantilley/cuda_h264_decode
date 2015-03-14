#ifndef V4L2_STREAM_H
#define V4L2_STREAM_H

#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define PIXEL_FMT            V4L2_PIX_FMT_H264
// #define

typedef struct _array_buf
{
	uint8_t* start;
	uint32_t length;
} array_buf;

typedef enum _ctrl_type
{
	V4L2stream_NULL,
	V4L2stream_EXPOSURE
} V4L2stream_ctrl_type;

typedef enum _change_type
{
	V4L2stream_ABSOLUTE,
	V4L2stream_RELATIVE
} V4L2stream_change_type;

class V4L2stream
{
public:

	V4L2stream( void );
	~V4L2stream( void );

	void init( uint32_t* width, uint32_t* height, std::string device_filename, uint32_t num_input_surfaces );

	void on( void );
	void off( void );

	int32_t changeControl( V4L2stream_ctrl_type, int32_t ctrl_value, V4L2stream_change_type );

	void getCodedFrame( int32_t ( * input_callback )( uint8_t* start, uint32_t payload_size ) );

private:

	int32_t     width;
	int32_t     height;
	std::string device;
	int32_t     numBufs;

	array_buf* buf_array;

	int32_t                    fd;
	struct v4l2_capability     device_caps;
	struct v4l2_format         format;
	struct v4l2_requestbuffers request_bufs;
	struct v4l2_buffer         buffer;
	struct v4l2_ext_controls   ext_ctrls;

	// ioctl wrapper, make inline for efficiency
	inline int32_t xioctl( int32_t file_desc, int32_t request, void* argp )
	{
		int32_t retVal;

		do retVal = ioctl( file_desc, request, argp );
		while( -1 == retVal && EINTR == errno );

		return retVal;
	}

};

#endif
