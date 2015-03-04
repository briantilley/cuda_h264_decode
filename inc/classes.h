// class declarations for BitPos, H264parser, and V4L2stream

#ifndef CLASSES_H
#define CLASSES_H

using std::string;

class BitPos
{
	public:

		BitPos( uint8_t* byte, uint8_t bitmask );
		~BitPos( void );

		uint8_t* getByte( void );
		void setByte( uint8_t* byte );
		uint8_t getMask( void );
		void setMask( uint8_t bitmask );

		uint32_t readBits( int32_t numBits );
		uint8_t readByte( void );

		bool readBitReverse( void );

	private:

		void advance( void );
		void retreat( void );

		uint8_t* byte;
		uint8_t  mask;

};

// H264parser

#include <nvcuvid.h> // only included here for friend functions

#include "RBSP_structs.h"

#define NAL_UNIT_START    0x000001
#define DEFAULT_SH_COUNT  4

class H264parser
{
	public:

		H264parser( BitPos starting_position );
		~H264parser( void );

		BitPos getPos( void );
		void setPos( BitPos position );

		void parseFrame( const uint8_t* start, uint32_t payload_size );

		// CUVIDPICPARAMS*  cuvidPicParams; // to be removed

	private:

		slice_header* makeSliceHeader( void );

		uint32_t uv  ( int32_t numBits );
		uint32_t uev ( void );
		int32_t  sev ( void );

		bool more_rbsp_data( void );

		void seqPmSet( uint8_t, uint8_t );
		void picPmSet( uint8_t, uint8_t );
		void sliceHeader( uint8_t, uint8_t );
		void refPicListMod( uint8_t, uint8_t );
		void predWeightTable( uint8_t, uint8_t );
		void decRefPicMark( uint8_t, uint8_t );

		uint8_t weightScale4x4[6][16];
		uint8_t weightScale8x8[2][64];
		bool defaultMatrix4x4[ 6 ];
		bool defaultMatrix8x8[ 2 ];
		void scaling_list( uint8_t*, uint8_t, bool* );

		BitPos pos;

		const uint8_t* start;
		uint32_t length;
		
		uint8_t nal_ref_idc;
		bool    idr_pic_flag;

		seq_param_set  SPS;
		pic_param_set  PPS;
		slice_header** SH;

		int32_t PrevFrameNum;

		uint32_t* SDOs;

		uint8_t SHidx;
		uint8_t maxSHcount;

	// I don't like having to do friend functions, but it's not a big problem
	friend int fillCuvidPicParams( H264parser*, CUVIDPICPARAMS* );
	friend int updateCuvidDPB( H264parser*, CUVIDPICPARAMS* );
	friend int clearCuvidDPB( CUVIDPICPARAMS* );
};

// V4L2stream

#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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

		V4L2stream( uint32_t width, uint32_t height, string device_filename, uint32_t num_input_surfaces );
		~V4L2stream( void );

		void init( void );

		void on( void );
		void off( void );

		int32_t changeControl( V4L2stream_ctrl_type, int32_t ctrl_value, V4L2stream_change_type );

		void getFrame( int32_t ( * input_callback )( uint8_t* start, uint32_t payload_size ) );

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

// CUVIDdecoder

#include <nvcuvid.h>

#define CODED_WIDTH       1920 // video dimensions should not be constants
#define CODED_HEIGHT      1088
#define TARGET_WIDTH      1920
#define TARGET_HEIGHT     1080

#define DECODE_SURFACES   8 // higher numbers = more memory usage
#define OUTPUT_SURFACES   8 // lower  numbers = possible slowdown

#define CUVID_CODEC       cudaVideoCodec_H264
#define CUVID_CHROMA      cudaVideoChromaFormat_422
#define CUVID_FLAGS       cudaVideoCreate_Default
#define CUVID_OUT_FORMAT  cudaVideoSurfaceFormat_NV12
#define CUVID_DEINTERLACE cudaVideoDeinterlaceMode_Adaptive

typedef enum _CUVIDdecoder_fmt
{
	CUVIDdecoder_NULL,
	CUVIDdecoder_H264
} CUVIDdecoder_fmt;

class CUVIDdecoder
{
	public:
		CUVIDdecoder( uint32_t width, uint32_t height, CUVIDdecoder_fmt, CUcontext* );
		~CUVIDdecoder( );

		// can't decide whether or not to split decode and map into two methods
		int32_t decodeFrame( CUVIDPICPARAMS*, int32_t ( * cuda_callback )( CUdeviceptr, uint32_t* p_mem_pitch ) );

	private:



};

#endif
