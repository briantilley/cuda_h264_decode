// This class is meant to simplify the display of CUDA images with openGL

#include <iostream>
#include <string.h>

#include "inc/constants.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

// cuda error checking
#define cudaErr(err) cudaError( err, __FILE__, __LINE__ )
inline void cudaError( cudaError_t err, const char* file, uint32_t line, bool abort=true )
{
    if( cudaSuccess != err )
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << cudaGetErrorName( err ) << endl;
        if( abort ) exit( err );
    }
}

// gl error checking
#define glErr( ) glError( glGetError( ), __FILE__, __LINE__ )
inline void glError( GLenum err, const char* file, uint32_t line, bool abort=false )
{
    if( GL_NO_ERROR != err )
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << glewGetErrorString( err ) << endl;
        if( abort ) exit( err );
    }
}

GLviewer::GLviewer( uint32_t width, uint32_t height, GLcolorMode mode )
{
	tex_pboWidth = width;
	tex_pboHeight = height;

	colorMode = mode;

	string vtxString, frgString;
	vtxString = loadTxtFileAsString( VERTEX_SOURCE_PATH );
	frgString = loadTxtFileAsString( FRAGMENT_SOURCE_PATH );

	const GLchar* vtxSrc = vtxString.c_str( );
	const GLchar* frgSrc = frgString.c_str( );

/////////////////////////////////////////////////

	glfwInit( );

	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, GL_VER_MAJ );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, GL_VER_MIN );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );

	#ifdef GLFW_RESIZEABLE
	glfwWindowHint( GLFW_RESIZABLE, GL_TRUE );
	#else
	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
	#endif

	#ifdef FULLSCREEN
	window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, glfwGetPrimaryMonitor( ), NULL);
	#else
	window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, NULL, NULL );
	#endif

	glfwMakeContextCurrent( window );

/////////////////////////////////////////////////

	glfwSetKeyCallback( window, ( GLFWkeyfun )&GLviewer::key_callback );
	glfwSetWindowSizeCallback( window, ( GLFWwindowsizefun )&GLviewer::windowSize_callback );
	glfwSetWindowCloseCallback( window, ( GLFWwindowclosefun )&GLviewer::windowClose_callback );

/////////////////////////////////////////////////

	glewExperimental = GL_TRUE;
	glewInit( );
	
	glErr( );

	// set cuda context
	cudaSetDevice( 0 );

/////////////////////////////////////////////////

	// set up vertices for texture
	GLfloat vertices[] = {
	//   X      Y     U     V
		-1.0f,  1.0f, 0.0f, 0.0f, // top left
		 1.0f,  1.0f, 1.0f, 0.0f, // top right
		-1.0f, -1.0f, 0.0f, 1.0f, // bottom left

		-1.0f, -1.0f, 0.0f, 1.0f, // bottom left
		 1.0f, -1.0f, 1.0f, 1.0f, // bottom right
		 1.0f,  1.0f, 1.0f, 0.0f // top right
	};

	GLbyte pboinit[ width * height * ( ( GLcolor_BW == mode ) ? 1 : 4 ) ];
	memset( pboinit, 0, sizeof( pboinit ) / 3 );
	memset( pboinit + sizeof( pboinit ) / 3, 127, sizeof( pboinit ) / 3 );
	memset( pboinit + 2 * sizeof( pboinit ) / 3, 255, sizeof( pboinit ) / 3 );

/////////////////////////////////////////////////

	glGenVertexArrays( 1, &vao );
	glBindVertexArray( vao );

	glGenBuffers( 1, &vbo );
	glBindBuffer( GL_ARRAY_BUFFER, vbo );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

	glGenBuffers( 1, &pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, width * height * ( ( GLcolor_BW == mode ) ? 1 : 4 ), pboinit, GL_STREAM_DRAW );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	// glEnable( GL_TEXTURE_2D );
	glGenTextures( 1, &tex );
	glBindTexture( GL_TEXTURE_2D, tex );
	glTexImage2D( GL_TEXTURE_2D, 0, ( ( GLcolor_BW == mode ) ? GL_RED : GL_RGBA ), width, height, 0, ( ( GLcolor_BW == mode ) ? GL_RED : GL_RGBA ), GL_UNSIGNED_BYTE, NULL );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	glErr( );

/////////////////////////////////////////////////

	cudaErr( cudaGraphicsGLRegisterBuffer( &cudaGfxPBO, pbo, cudaGraphicsRegisterFlagsNone ) );

/////////////////////////////////////////////////

	vtxShd = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vtxShd, 1, &vtxSrc, NULL );
	glCompileShader( vtxShd );

	frgShd = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( frgShd, 1, &frgSrc, NULL );
	glCompileShader( frgShd );

	checkShaderCompile( );

	shaders = glCreateProgram( );
	glAttachShader( shaders, vtxShd );
	glAttachShader( shaders, frgShd );

	glBindFragDataLocation( shaders, 0, "outColor" );

	glLinkProgram( shaders );
	glUseProgram( shaders );

	glErr( );

/////////////////////////////////////////////////

	GLuint posAtt = glGetAttribLocation( shaders, "position" );
	glEnableVertexAttribArray( posAtt );
	glVertexAttribPointer( posAtt, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( GLfloat ), 0 );

	GLuint texAtt = glGetAttribLocation( shaders, "texcoord" );
	glEnableVertexAttribArray( texAtt );
	glVertexAttribPointer( texAtt, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( GLfloat ), ( const GLvoid* )( 2 * sizeof( GLfloat ) ) );

	glErr( );

/////////////////////////////////////////////////

	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
}

GLviewer::~GLviewer( void )
{
	glDeleteBuffers( 1, &pbo );
	glDeleteBuffers( 1, &vbo );
	glDeleteTextures( 1, &tex );

	glDeleteProgram( shaders );
	glDeleteShader( frgShd );
	glDeleteShader( vtxShd );

	glfwDestroyWindow( window );
}

int32_t GLviewer::mapOutputImage( uint8_t** pDest )
{
	size_t size; // trash it

	cudaErr( cudaGraphicsMapResources( 1, &cudaGfxPBO, 0 ) );
	cudaErr( cudaGraphicsResourceGetMappedPointer( ( void** )pDest, &size, cudaGfxPBO ) );

	return 0;
}

int32_t GLviewer::unmapOutputImage( void )
{
	cudaErr( cudaGraphicsUnmapResources( 1, &cudaGfxPBO, 0 ) );

	return 0;
}

// rework to use proper openGL
int32_t GLviewer::display( void )
{
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBindTexture( GL_TEXTURE_2D, tex );

	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, tex_pboWidth, tex_pboHeight, ( ( GLcolor_BW == colorMode ) ? GL_RED : GL_RGBA ), GL_UNSIGNED_BYTE, NULL );

	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	glClear( GL_COLOR_BUFFER_BIT );
	glDrawArrays( GL_TRIANGLES, 0, 6 );
	glfwSwapBuffers( window );
	glfwPollEvents( );

	glErr( );

	return 0;
}

void GLviewer::checkShaderCompile( void )
{
	GLint status;

	glGetShaderiv( vtxShd, GL_COMPILE_STATUS, &status );

	if( GL_TRUE != status )
	{
		char buffer[ 512 ];
		glGetShaderInfoLog( vtxShd, 512, NULL, buffer );
		std::cerr << "vert compile fail: " << buffer << endl;
	}

	glGetShaderiv( frgShd, GL_COMPILE_STATUS, &status );

	if( GL_TRUE != status )
	{
		char buffer[ 512 ];
		glGetShaderInfoLog( frgShd, 512, NULL, buffer );
		std::cerr << "frag compile fail: " << buffer << endl;
	}
}

// GLFW callbacks (static, as GLFW is a C lib)

void GLviewer::key_callback( GLFWwindow* cb_window, int key, int scancode, int action, int mods )
{
	// printf( "[keypress] key: %d scancode: %d action: %d mods: %d\n", key, scancode, action, mods );

	if( GLFW_PRESS == action )
	{
		switch( key )
		{
			case GLFW_KEY_ESCAPE:

				#ifdef FULLSCREEN
				cleanUp( ); // end the program
				exit( 0 );
				#endif
			
			break;

			case GLFW_KEY_F4:

				if( GLFW_PRESS == glfwGetKey( cb_window, GLFW_KEY_LEFT_ALT ) || GLFW_PRESS == glfwGetKey( cb_window, GLFW_KEY_RIGHT_ALT ) )
					cleanUp( );
					exit( 0 );
			
			default:
			break;
		}
	}
	else if( GLFW_RELEASE == action )
	{

	}
	else if( GLFW_REPEAT == action )
	{

	}
}

void GLviewer::windowSize_callback( GLFWwindow* cb_window, int width, int height )
{
	// I think this does a costly framebuffer reallocation on every call.
	// Find a way to avoid that.
	glViewport( 0, 0, width, height );
}

void GLviewer::windowClose_callback( GLFWwindow* cb_window )
{
	#ifndef FULLSCREEN
	cleanUp( );
	exit( 0 );
	#endif
}