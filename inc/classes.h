#ifndef CLASSES_H
#define CLASSES_H

class BitPos
{
	public:
		
		BitPos( );
		BitPos( uint8_t* byte );
		BitPos( uint8_t* byte, uint8_t mask );
		~BitPos( );

		void advance(void);

	private:

		uint8_t* byte;
		uint8_t  mask;
};

#include <nvcuvid.h>
#include "RBSP_structs.h"

class H264parser
{
	public:

		H264parser( );
		H264parser( BitPos pos );

		void update( BitPos pos );

		void populate( CUVIDPICPARAMS &PicParams );

	private:

		BitPos* pPos;
		
		seq_param_set SeqParamSet;
		pic_param_set PicParamSet;
		slice_header  SliceHeader;
};

#endif