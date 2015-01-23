#include <string>
#include <iostream>
#include <nvcuvid.h>

#include "inc/RBSP_structs.h"

class BitPos;

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