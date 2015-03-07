// This class is meant to simplify the display of CUDA images with openGL

#include <iostream>

#include "inc/constants.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

GLviewer::GLviewer( uint32_t width, uint32_t height, GLcolorMode mode )
{
	tex_pboWidth = width;
	tex_pboHeight = height;

	colorMode = mode;

	glfwInit( );

	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, GL_VER_MAJ );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, GL_VER_MIN );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );

	#ifdef GLFW_RESIZEABLE
	std::cerr << "not ready for resizable windows, yet!" << endl;
	exit( 1 );
	#else
	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
	#endif

	#ifdef FULLSCREEN
	window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, glfwGetPrimaryMonitor( ), NULL);
	#else
	window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, NULL, NULL );
	#endif

	glfwMakeContextCurrent( window );

	glewExperimental = GL_TRUE;
	glewInit( );

	glClearColor( 0.0, 0.0, 0.0, 1.0 );

	// clear both buffers to black
	glClear( GL_COLOR_BUFFER_BIT );
	glfwSwapBuffers( window );
	glClear( GL_COLOR_BUFFER_BIT );
	glfwSwapBuffers( window );

	// allocate pixel unpack buffer
	glGenBuffers( 1, &pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, width * height * ( ( GLcolor_BW == mode ) ? 1 : 4 ), NULL, GL_STREAM_DRAW );
	cudaGLRegisterBufferObject( pbo ); // let CUDA and GL agree on this buffer

	// allocate texture
	glEnable( GL_TEXTURE_2D );
	glGenTextures( 1, &tex );
	glBindTexture( GL_TEXTURE_2D, tex );
	glTexImage2D( GL_TEXTURE_2D,
		0,
		( ( GLcolor_BW == mode ) ? GL_LUMINANCE : GL_RGBA ),
		width, height,
		0,
		( ( GLcolor_BW == mode ) ? GL_LUMINANCE : GL_RGBA ),
		GL_UNSIGNED_BYTE,
		NULL );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
}

GLviewer::~GLviewer( void )
{
	glDeleteBuffers( 1, &pbo );
	glDeleteTextures( 1, &tex );

	glfwDestroyWindow( window );
}

int32_t GLviewer::writeToOutput( int32_t ( * GPUwrite_callback )( uint8_t*, uint32_t, uint32_t, bool ) )
{
	uint8_t* dPBO;

	cudaGLMapBufferObject( ( void** )&dPBO, pbo );

	GPUwrite_callback( dPBO, tex_pboWidth, tex_pboHeight, ( ( GLcolor_RGBA == colorMode ) ? true : false ) );

	cudaGLUnmapBufferObject( pbo );

	return 0;
}

int32_t GLviewer::mapOutputImage( uint8_t** pDest )
{
	cudaGLMapBufferObject( ( void** )pDest, pbo );
}

int32_t GLviewer::unmapOutputImage( void )
{
	cudaGLUnmapBufferObject( pbo );
}

int32_t GLviewer::display( void )
{
	// may not be needed
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBindTexture( GL_TEXTURE_2D, tex );

	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, tex_pboWidth, tex_pboHeight,
		( ( GLcolor_BW == colorMode ) ? GL_LUMINANCE : GL_RGBA ),
		GL_UNSIGNED_BYTE, NULL );

	// removed from core GL
	glBegin( GL_QUADS );
		glTexCoord2f( 0.0f, 1.0f );
		glVertex3f( 0.0f, 0.0f, 0.0 );

		glTexCoord2f( 0.0f, 0.0f );
		glVertex3f( 0.0f, 1.0f, 0.0 );

		glTexCoord2f( 1.0f, 0.0f );
		glVertex3f( 1.0f, 1.0f, 0.0 );

		glTexCoord2f( 1.0f, 1.0f );
		glVertex3f( 1.0f, 0.0f, 0.0 );
	glEnd( );

//	glDrawArrays( GL_TRIANGLES, 0, 6 );
	glfwSwapBuffers( window );
	glfwPollEvents( );

	return 0;
}