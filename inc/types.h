#ifndef TYPES_H
#define TYPES_H

typedef enum _nal_type
{
	SEQ_PM_SET,
	PIC_PM_SET,
	SLICE_HEAD
} nal_type;

typedef struct _array_buf
{
	uint8_t* start;
	uint32_t length;
} array_buf;

#endif