# compilers
CUDA=nvcc
STD=g++

# filenames
OUTPUT=out

# headers
INC=-I/usr/local/cuda-6.5/include

# libraries
LIB_PATHS=-L/usr/lib/nvidia-340

# flags
FLAGS=-std=gnu++11
CUFLAGS=-std=c++11

# all linked objects
MASTER_LINKS=-lcuda -lcudart -lnvcuvid -lglfw -lGLEW -lGL

# targets (excluding parser to try to use nvidia's)
all: v4l2 cuvid gl cuda
	$(CUDA) main.cpp `ls *.o` -o $(OUTPUT) $(MASTER_LINKS) $(INC) $(LIB_PATHS) $(CUFLAGS)

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(FLAGS) $(INC)

cuvid: CUVIDparser.cpp
	$(STD) CUVIDparser.cpp -c -lnvcuvid -lcuda $(INC) $(LIB_PATHS) $(FLAGS)

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
