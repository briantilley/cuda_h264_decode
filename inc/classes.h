#ifndef CLASSES_H
#define CLASSES_H

#include <stdint.h>
#include <string>

using std::string;

class BitPos
{
	public:
		
		BitPos( void );
		BitPos( uint8_t* );
		BitPos( uint8_t*, uint8_t );
		~BitPos( void );

		uint8_t* getByte( void );
		void setByte( uint8_t* );
		uint8_t getMask( void );
		void setMask( uint8_t  );

		void advance( void );
		uint32_t readBits( int32_t numBits );

	private:

		uint8_t* byte;
		uint8_t  mask;

};

#include <nvcuvid.h>
#include "RBSP_structs.h"

class H264parser
{
	public:

		H264parser( void );
		H264parser( BitPos );
		~H264parser( void );

		BitPos getPos( void );
		void setPos( BitPos );

		// return type? seek( void ), seek( BitPos )

		void update( nal_type );
		void update( BitPos, nal_type );

		void populate( CUVIDPICPARAMS& );

		bool     flag( void );
		uint32_t uv  ( int32_t );
		uint32_t uev ( void );
		int32_t  sev ( void );

	private:

		BitPos pos;
		
		seq_param_set SPS;
		pic_param_set PPS;
		slice_header  SH;

};

#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

class V4L2stream
{
	public:

		V4L2stream( void );
		V4L2stream( int, int, string, int );
		~V4L2stream( void );
	
		void setWidth( int32_t );
		int32_t getWidth( void );

		void setHeight( int32_t );
		int32_t getHeight( void );
		
		void setDevice( string );
		string getDevice( void );

		void setBufs( int32_t );
		int32_t getBufs( void );

		void initialize( void );

		void on( void );
		void off( void );

		void getFrame( uint8_t* ); // alloc input ptr before using

	private:

		int32_t xioctl( int32_t, int32_t, void* );

		int32_t width;
		int32_t height;
		string  device;
		int32_t numBufs;

		array_buf* buf_array;

		int32_t                    fd;
		struct v4l2_capability     device_caps;
		struct v4l2_format         format;
		struct v4l2_requestbuffers request_bufs;
		struct v4l2_buffer         buffer;
		struct v4l2_ext_controls   ext_ctrls;

};

#endif