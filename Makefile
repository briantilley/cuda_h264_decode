# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=out
OBJECTS=V4L2stream.o BitPos.o H264parser.o CUVIDdecoder.o GLviewer.o cuda.o

#headers
INC=-I/usr/local/cuda-6.5/include

#libraries
LIB_PATHS=-L/usr/lib/nvidia-340

#flags
FLAGS=-std=gnu++11
CUFLAGS=-std=c++11 -w

GL_LINKS=-lglfw -lGLEW -lGL

#targets
all: classes main
	@# runs classes and main targets in sequence
	@# default behavior is to compile the entire program

# compile only main.cpp, then link preexisting object files from other sources
main: $(MAIN)
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) -lnvcuvid -lcuda -lcudart $(CUFLAGS) $(GL_LINKS)

# generate object files from all source files sans main.cpp
classes: v4l2 bitpos parser cuvid cuda gl
	@# will automatically make targets

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(INC) $(FLAGS)

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(INC) $(FLAGS)

parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(INC) $(FLAGS)

cuvid: CUVIDdecoder.cpp
	$(CUDA) CUVIDdecoder.cpp -c -lnvcuvid -lcuda $(INC) $(LIB_PATHS) $(CUFLAGS)

gl: GLviewer.cpp
	$(CUDA) GLviewer.cpp -c -lcudart $(GL_LINKS) $(CUFLAGS)

cuda: cuda.cu
	$(CUDA) cuda.cu -c -lcudart $(CUFLAGS)

# run target is for convenience, executes the program
run: $(OUTPUT)
	./$(OUTPUT)

# remove the executable and all object files (useful for svc, e.g. git commits)
clean:
	rm $(OUTPUT) $(OBJECTS)
