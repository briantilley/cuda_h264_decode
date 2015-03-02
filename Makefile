# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=out
OBJECTS=BitPos.o H264parser.o V4L2stream.o NALparser.o cuvidHandler.o cuda.o

#headers
INC=-I/usr/local/cuda-6.5/include

#libraries
CUDA_LIBS=-lnvcuvid -lcuda
LIB_PATHS=-L/usr/lib/nvidia-340

#flags
FLAGS=-std=gnu++11
CUFLAGS=-std=c++11

#targets
all: classes main
	@# runs classes and main targets in sequence

main: $(MAIN)
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) $(CUDA_LIBS) $(CUFLAGS)

classes: bitpos parser v4l2 nalparser cuvid cuda
	@# will automatically make targets

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(INC) $(FLAGS)

parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(INC) $(FLAGS)

nalparser: NALparser.cpp
	$(STD) NALparser.cpp -c $(INC) $(FLAGS)

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(INC) $(FLAGS)

cuvid: cuvidHandler.cpp
	$(STD) cuvidHandler.cpp -c $(INC) $(FLAGS)

cuda: cuda.cu
	$(CUDA) cuda.cu -c $(INC) $(CUFLAGS) $(LIB_PATHS) $(CUDA_LIBS)

run: $(OUTPUT)
	./$(OUTPUT)

clean:
	rm $(OUTPUT) $(OBJECTS)
