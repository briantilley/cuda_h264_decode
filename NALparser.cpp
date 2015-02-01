#include <string>
#include <iostream>
#include <nvcuvid.h>

#include "inc/constants.h"
#include "inc/RBSP_structs.h"
#include "inc/types.h"
#include "inc/classes.h"

using std::cout;
using std::endl;
using std::string;

void H264parser::seqPmSet( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::picPmSet( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::sliceHeader( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::refPicListMod( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::predWeightTable( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}

void H264parser::decRefPicMark( uint8_t nal_ref_idc, uint8_t nal_type )
{
	// ...
}