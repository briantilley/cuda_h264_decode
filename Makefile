# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=out
OBJECTS=BitPos.o H264parser.o V4L2stream.o NALparser.o cuvidHandler.o

#headers
INC=-I/usr/local/cuda-6.5/include

#libraries
LIBS=-lnvcuvid -lcuda -lcudart
LIB_PATHS=-L/usr/lib/nvidia-340

#flags
FLAGS=-std=gnu++11
CUFLAGS=-std=c++11

#targets
all: classes main
	@# runs classes and main targets in sequence

main: $(MAIN)
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) $(LIBS) $(CUFLAGS)

classes: bitpos parser v4l2 nalparser cuvid
	@# will automatically make targets

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(INC) $(LIB_PATHS) $(LIBS) $(FLAGS)

parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(INC) $(LIB_PATHS) $(LIBS) $(FLAGS)

nalparser: NALparser.cpp
	$(STD) NALparser.cpp -c $(INC) $(LIB_PATHS) $(LIBS) $(FLAGS)

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(INC) $(LIB_PATHS) $(LIBS) $(FLAGS)

cuvid: cuvidHandler.cpp
	$(STD) cuvidHandler.cpp -c $(INC) $(LIB_PATHS) $(LIBS) $(FLAGS)

run: $(OUTPUT)
	@./$(OUTPUT)

clean:
	rm $(OUTPUT) $(OBJECTS)