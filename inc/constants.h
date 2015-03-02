#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NAL_UNIT_START    0x000001

#define CODED_WIDTH       1920
#define CODED_HEIGHT      1088
#define TARGET_WIDTH      1920
#define TARGET_HEIGHT     1080

#define DECODE_SURFACES   8
#define DISPLAY_SURFACES  8

#define CUVID_CODEC       cudaVideoCodec_H264
#define CUVID_CHROMA      cudaVideoChromaFormat_422
#define CUVID_FLAGS       cudaVideoCreate_Default
#define CUVID_OUT_FORMAT  cudaVideoSurfaceFormat_NV12
#define CUVID_DEINTERLACE cudaVideoDeinterlaceMode_Adaptive

#define DEFAULT_DEVICE    "/dev/video0"
#define DEFAULT_NUMBUFS   8

#define DEFAULT_SH_COUNT  4

#define DPB_SIZE          16

#endif