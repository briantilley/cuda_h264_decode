// This class is meant to simplify the display of CUDA images with openGL

#include <iostream>
#include <cmath>

#include "inc/constants.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

GLviewer::GLviewer( uint32_t width, uint32_t height )
{
	glfwInit( );

	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, GL_VER_MAJ );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, GL_VER_MIN );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );

	#ifdef GLFW_RESIZABLE
	std::cerr << "not ready for resizable windows, yet!" << endl;
	exit( 1 );
	#else
	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
	#endif

	glewExperimental = GL_TRUE;
	glewInit( );

}

GLviewer::~GLviewer( void )
{
	if( 0 != pbo ) glDeleteBuffers( 1, &pbo );
	if( 0 != vbo ) glDeleteBuffers( 1, &vbo );
	if( 0 != tex ) glDeleteTextures( 1, &tex );

	// figure out shader deletion
}

int32_t GLviewer::openWindow( void )
{
	#ifdef FULLSCREEN
	*window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, glfwGetPrimaryMonitor( ), NULL)
	#else
	*window = glfwCreateWindow( WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, NULL, NULL );
	#endif

	glfwMakeContextCurrent( window );

	return 0;
}

int32_t GLviewer::closeWindow( void )
{
	glfwMakeContextCurrent( NULL );
}