// This class is meant to simplify the display of CUDA images with openGL
#define GLEW_STATIC

// C++
#include <iostream>
#include <cstring> // for memset()

// header
#include "inc/GLviewer.h"

using std::cout;
using std::endl;
using std::string;

// cuda error checking
#define cudaErr(err) cudaError( err, __FILE__, __LINE__ )
inline void cudaError( cudaError_t err, const char file[], uint32_t line, bool abort=true )
{
    if( cudaSuccess != err )
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << cudaGetErrorName( err ) << endl;
        if( abort ) exit( err );
    }
}

// gl error checking
// don't set abort to true unless you absolutely need to
#define glErr( ) glError( glGetError( ), __FILE__, __LINE__ )
inline void glError( GLenum err, const char file[], uint32_t line, bool abort=false )
{
    if( GL_NO_ERROR != err )
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << glewGetErrorString( err ) << endl;
        if( abort ) exit( err );
    }
}

// static initializations
void ( * GLviewer::pfnAppExit )( void ) = NULL;
bool GLviewer::fullscreen = false;
bool GLviewer::color = false;

GLviewer::GLviewer( uint32_t in_texWidth, uint32_t in_texHeight, uint32_t in_windowWidth, uint32_t in_windowHeight, uint32_t flags, void ( * in_fn_appExit )( void ) )
{
	// make use of arguments
	texWidth = in_texWidth;
	texHeight = in_texHeight;
	windowWidth = in_windowWidth;
	windowHeight = in_windowHeight;

	fullscreen = ( 0 != ( flags & GLviewer_fullscreen ) );
	color = ( 0 != ( flags & GLviewer_color ) );

	pfnAppExit = in_fn_appExit;

	// load shader source files into GLchar arrays
	string vtxString, frgString;
	vtxString = loadTxtFileAsString( VERTEX_SOURCE_PATH );
	frgString = loadTxtFileAsString( FRAGMENT_SOURCE_PATH );

	const GLchar* const vtxSrc = vtxString.c_str( );
	const GLchar* const frgSrc = frgString.c_str( );

/////////////////////////////////////////////////

	// set up GLFW openGL context handling library
	glfwInit( );

	// set openGL version and prevent use of deprecated openGL functionality
	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, GL_VER_MAJ );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, GL_VER_MIN );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );

	// option to set whether window can be resized (setting to false may have adverse effects in fullscreen)
	#ifdef GLFW_RESIZEABLE
	glfwWindowHint( GLFW_RESIZABLE, GL_TRUE );
	#else
	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
	#endif

	// open the window
	if( fullscreen )
		window = glfwCreateWindow( windowWidth, windowHeight, WINDOW_NAME, glfwGetPrimaryMonitor( ), NULL);
	else
		window = glfwCreateWindow( windowWidth, windowHeight, WINDOW_NAME, NULL, NULL );

	glfwMakeContextCurrent( window );

/////////////////////////////////////////////////

	// tell GLFW about the event callbacks
	glfwSetKeyCallback( window, ( GLFWkeyfun )&GLviewer::key_callback );
	glfwSetWindowSizeCallback( window, ( GLFWwindowsizefun )&GLviewer::windowSize_callback );
	glfwSetWindowCloseCallback( window, ( GLFWwindowclosefun )&GLviewer::windowClose_callback );

/////////////////////////////////////////////////

	glewExperimental = GL_TRUE; // force GLEW to use latest GL functionality
	glewInit( ); // expected to cause "Unknown Error"
	glErr( );

	// explicitly open cuda context
	cudaSetDevice( 0 );

/////////////////////////////////////////////////

	// set up default vertices for texture
	GLfloat vertices[] = {
	//   X      Y     U     V
		-1.0f,  1.0f, 0.0f, 0.0f, // top left
		 1.0f,  1.0f, 1.0f, 0.0f, // top right
		-1.0f, -1.0f, 0.0f, 1.0f, // bottom left

		-1.0f, -1.0f, 0.0f, 1.0f, // bottom left
		 1.0f, -1.0f, 1.0f, 1.0f, // bottom right
		 1.0f,  1.0f, 1.0f, 0.0f  // top right
	};

/////////////////////////////////////////////////

	// initialize buffers & texture
	glGenVertexArrays( 1, &vao );
	glBindVertexArray( vao );

	glGenBuffers( 1, &vbo );
	glBindBuffer( GL_ARRAY_BUFFER, vbo );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

	glGenBuffers( 1, &pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, texWidth * texHeight * ( color ? 4 : 1 ), NULL, GL_STREAM_DRAW );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	glGenTextures( 1, &tex );
	glBindTexture( GL_TEXTURE_2D, tex );
	glTexImage2D( GL_TEXTURE_2D, 0, ( color ? GL_RGBA : GL_RED ), texWidth, texHeight, 0, ( color ? GL_RGBA : GL_RED ), GL_UNSIGNED_BYTE, NULL );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	glErr( );

/////////////////////////////////////////////////

	// get a CUDA graphics resource to the PBO
	cudaErr( cudaGraphicsGLRegisterBuffer( &cudaGfxPBO, pbo, cudaGraphicsRegisterFlagsNone ) );

/////////////////////////////////////////////////

	// compile shader source strings into a shader program
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

	// implicit in openGL if this is the only "out" in frag source
	glBindFragDataLocation( shaders, 0, "outColor" );

	glLinkProgram( shaders );
	glUseProgram( shaders );

	glErr( );

/////////////////////////////////////////////////

	// inform openGL about vertex buffer format
	GLuint posAtt = glGetAttribLocation( shaders, "position" );
	glEnableVertexAttribArray( posAtt );
	glVertexAttribPointer( posAtt, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( GLfloat ), 0 );

	GLuint texAtt = glGetAttribLocation( shaders, "texcoord" );
	glEnableVertexAttribArray( texAtt );
	glVertexAttribPointer( texAtt, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( GLfloat ), ( const GLvoid* )( 2 * sizeof( GLfloat ) ) );

	glErr( );

/////////////////////////////////////////////////

	// set black as "blank" color
	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
}

GLviewer::~GLviewer( void )
{
	// must unregister from CUDA before deleting from openGL
	cudaErr( cudaGraphicsUnregisterResource( cudaGfxPBO ) );

	glDeleteBuffers( 1, &pbo );
	glDeleteBuffers( 1, &vbo );
	glDeleteTextures( 1, &tex );

	glDeleteProgram( shaders );
	glDeleteShader( frgShd );
	glDeleteShader( vtxShd );

	glfwDestroyWindow( window );
}

int32_t GLviewer::mapDispImage( void** pDest )
{
	// don't send back the size of the mapped buffer
	size_t trash;

	// map PBO from openGL and get a CUDA device pointer to it
	cudaErr( cudaGraphicsMapResources( 1, &cudaGfxPBO, 0 ) );
	cudaErr( cudaGraphicsResourceGetMappedPointer( pDest, &trash, cudaGfxPBO ) );

	return 0;
}

int32_t GLviewer::unmapDispImage( void )
{
	// release PBO to openGL
	cudaErr( cudaGraphicsUnmapResources( 1, &cudaGfxPBO, 0 ) );

	return 0;
}

// rework to use proper openGL
int32_t GLviewer::display( void )
{
	// bind texture and PBO
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBindTexture( GL_TEXTURE_2D, tex );

	// copy PBO to texture
	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, ( color ? GL_RGBA : GL_RED ), GL_UNSIGNED_BYTE, NULL );

	// unbind PBO (texture needed for glDrawArrays)
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	// main draw
	glClear( GL_COLOR_BUFFER_BIT );
	glDrawArrays( GL_TRIANGLES, 0, 6 );
	glfwSwapBuffers( window );

	// finally unbind texture
	glBindTexture( GL_TEXTURE_2D, 0 );

	glErr( );

	return 0;
}

// call this method in application's main loop to respond to input
int32_t GLviewer::loop( void )
{
	glfwPollEvents( );

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

				// end on ESC in full screen
				if( fullscreen ) pfnAppExit( );
			
			break;

			case GLFW_KEY_F4:

				// end on alt+f4 always
				if( GLFW_PRESS == glfwGetKey( cb_window, GLFW_KEY_LEFT_ALT ) || GLFW_PRESS == glfwGetKey( cb_window, GLFW_KEY_RIGHT_ALT ) )
					pfnAppExit( );
			
			default:
			break;
		}
	}
	// else if( GLFW_RELEASE == action )
	// {

	// }
	// else if( GLFW_REPEAT == action )
	// {

	// }
}

// need to call the instance attached to cb_window
void GLviewer::windowSize_callback( GLFWwindow* cb_window, int width, int height )
{
	// I think this does a costly framebuffer reallocation on every call.
	// Find a way to avoid that.
	glViewport( 0, 0, width, height );
}

void GLviewer::windowClose_callback( GLFWwindow* cb_window )
{
	pfnAppExit( );
}