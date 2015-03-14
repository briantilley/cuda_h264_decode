#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// main

void appEnd( void );
std::string loadTxtFileAsString( const char[] );

// cuda

int32_t NV12toRGBA( const uint8_t*, uint8_t**, const uint32_t, uint32_t*, const uint32_t, const uint32_t );
int32_t NV12toBW  ( const uint8_t*, uint8_t**, const uint32_t, uint32_t*, const uint32_t, const uint32_t );

#endif