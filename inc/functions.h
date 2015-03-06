#ifndef FUNCTIONS_H
#define FUNCTIONS_H

int32_t coded_callback( uint8_t* start, uint32_t length );
int32_t decoded_callback( const CUdeviceptr devPtr, uint32_t pitch );

void cleanUp( void );

std::string loadTxtFileAsString( const char* );

int fillCuvidPicParams( H264parser* parser, CUVIDPICPARAMS* params );
int updateCuvidDPB( H264parser* parser, CUVIDPICPARAMS* params );
int clearCuvidDPB( CUVIDPICPARAMS* params );

#endif