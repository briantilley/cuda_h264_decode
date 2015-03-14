// all magic values should end up in defines here
// to centralize compile-time constants

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define DEVICE         "/dev/video0"
#define WIDTH          1920
#define HEIGHT         1080

#define INPUT_SURFACES 8

#ifndef DECODE_SURFACES
    #define DECODE_SURFACES   8
#endif

#ifndef OUTPUT_SURFACES
    #define OUTPUT_SURFACES   8
#endif

#endif
