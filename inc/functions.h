#ifndef FUNCTIONS_H
#define FUNCTIONS_H value

int fillCuvidPicParams( H264parser* parser, CUVIDPICPARAMS* params );
int updateCuvidDPB( H264parser* parser, CUVIDPICPARAMS* params );
int clearCuvidDPB( CUVIDPICPARAMS* params );

void cleanUp( void );

#endif