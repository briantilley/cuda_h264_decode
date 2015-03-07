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

#include <cuviddec.h> // only included here for friend functions

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

	void getCodedFrame( int32_t ( * input_callback )( uint8_t* start, uint32_t payload_size ) );

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

#define DECODE_GAP        0 // number of frames decode should be ahead of map

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

	CUVIDdecoder( uint32_t width, uint32_t height, CUVIDdecoder_fmt );
	~CUVIDdecoder( );

	// can't decide whether or not to split decode and map into two methods
	int32_t getDecodedFrame( CUVIDPICPARAMS*, int32_t ( * cuda_callback )( const CUdeviceptr, uint32_t pitch ) );

private:

	CUvideodecoder         decoder;
	CUVIDDECODECREATEINFO* pdci;

};

// GLviewer

#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#define GL_VER_MAJ 3 // openGL version 3.2
#define GL_VER_MIN 2

#define WINDOW_NAME "decode2"

// #define FULLSCREEN

#ifdef FULLSCREEN //---
#define WINDOW_WIDTH  1920
#define WINDOW_HEIGHT 1080
#else //---
#define WINDOW_WIDTH  640
#define WINDOW_HEIGHT 480
#endif //---

#ifdef GLFW_RESIZEABLE // not worrying about resizing now
#undef GLFW_RESIZEABLE
#endif

#define MAX_SHADERS   2

typedef enum _GLcolorMode
{
	GLcolor_BW,
	GLcolor_RGBA
} GLcolorMode;

class GLviewer
{
public:

	GLviewer( uint32_t width, uint32_t height, GLcolorMode );
	~GLviewer( void );

	int32_t writeToOutput( int32_t ( * )( uint8_t*, uint32_t, uint32_t, bool ) );

	int32_t mapOutputImage( uint8_t** );
	int32_t unmapOutputImage( void );

	int32_t display( void );

private:

	GLFWwindow* window;
	
	GLuint pbo; // pixel buffer object - the buffer CUDA writes images to
	GLuint tex; // texture - what GL renders

	GLcolorMode colorMode;

	uint32_t tex_pboWidth;
	uint32_t tex_pboHeight;

};

#endif
