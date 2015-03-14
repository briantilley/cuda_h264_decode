# compilers
CUDA=nvcc
STD=g++

# filenames
OUTPUT=out
OBJECTS=V4L2stream.o BitPos.o H264parser.o CUVIDdecoder.o GLviewer.o cuda.o

# headers
INC=-I/usr/local/cuda-6.5/include

# libraries
LIB_PATHS=-L/usr/lib/nvidia-340

# flags
FLAGS=-std=gnu++11
CUFLAGS=-std=c++11 -w

# all linked objects
MASTER_LINKS=-lcuda -lcudart -lnvcuvid -lglfw -lGLEW -lGL

# targets (excluding parser to try to use nvidia's)
all: v4l2 bitpos cuvid gl cuda
	$(CUDA) main.cpp `ls *.o` -o $(OUTPUT) $(MASTER_LINKS) $(INC) $(LIB_PATHS) $(CUFLAGS)

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(FLAGS) $(INC)

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(FLAGS) $(INC)

# not in use
parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(FLAGS) $(INC)

cuvid: CUVIDdecoder.cpp
	$(STD) CUVIDdecoder.cpp -c -lnvcuvid -lcuda $(INC) $(LIB_PATHS) $(FLAGS)

gl: GLviewer.cpp
	$(STD) GLviewer.cpp -c -lcudart -lglfw -lGLEW -lGL $(FLAGS) $(INC)

cuda: cuda.cu
	$(CUDA) cuda.cu -c -lcudart $(CUFLAGS) $(INC)

# run target is for convenience, executes the program
run: $(OUTPUT)
	./$(OUTPUT)

# remove the executable and all object files (useful for svc, e.g. git commits)
clean:
	rm $(OUTPUT) `ls *.o` core
