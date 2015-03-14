// I'm considering making this whole freaking class static

#ifndef GL_VIEWER_H
#define GL_VIEWER_H

// openGL dependencies
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define GL_VER_MAJ 3 // openGL version 3.2
#define GL_VER_MIN 2

#define WINDOW_NAME "decode2"

#define GLFW_RESIZEABLE

#define VERTEX_SOURCE_PATH   "shaders/vert.glsl"
#define FRAGMENT_SOURCE_PATH "shaders/frag.glsl"

typedef enum _GLviewerCreateFlags
{
	GLviewer_none       = 0x00,
	GLviewer_fullscreen = 0x01,
	GLviewer_color      = 0x02
} GLviewerCreateFlags;

// needed function from main
std::string loadTxtFileAsString( const char[] );

class GLviewer
{
public:

	// main application exit function
	// must be declared static to be called from GLFW callback (not ideal)
	static void ( * pfnAppExit )( void );

	GLviewer( uint32_t texWidth, uint32_t texHeight, uint32_t windowWidth, uint32_t windowHeight, uint32_t GLviewerCreateFlags, void ( * )( void ) );
	~GLviewer( void );

	// map the destination device pointer
	// write image to destination in CUDA
	// unmap the image
	// display it
	// NOTE: openGL only draws when display is called
	int32_t mapDispImage( void** );
	int32_t unmapDispImage( void );
	int32_t display( void );

	// call this method in the application's main loop to respond to input
	int32_t loop( void );

	// creation flags
	static bool fullscreen, color;

private:
	
	// CUDA refs
	cudaGraphicsResource_t cudaGfxPBO;

	// GL refs
	GLFWwindow* window; // main context
	GLuint pbo; // pixel buffer object - the buffer CUDA writes images to
	GLuint tex; // texture - what GL renders
	GLuint vbo; // vertices - texture destination

	GLuint vao; // vertex attrib array

	GLuint vtxShd, frgShd; // shaders
		GLuint shaders; // shader program

	// texture and window widths;
	uint32_t texWidth, texHeight;
	uint32_t windowWidth, windowHeight;

	void checkShaderCompile( void );

	// static because GLFW is a C library
	static void key_callback( GLFWwindow*, int key, int scancode, int action, int mods );
	static void windowSize_callback( GLFWwindow*, int width, int height );
	static void windowClose_callback( GLFWwindow* );

};

#endif