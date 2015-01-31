# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=out
OBJECTS=BitPos.o H264parser.o V4L2stream.o

#headers
INC=-I/usr/local/cuda-6.5/include

#libraries
LIBS=-lnvcuvid -lcuda -lcudart
LIB_PATHS=-L/usr/lib/nvidia-340

#targets
all: $(MAIN) classes
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) $(LIBS)

main: $(MAIN)
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) $(LIBS)

classes: bitpos parser v4l2
	@# will automatically make targets

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(INC) $(LIB_PATHS) $(LIBS)

parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(INC) $(LIB_PATHS) $(LIBS)

v4l2: V4L2stream.cpp
	$(STD) V4L2stream.cpp -c $(INC) $(LIB_PATHS) $(LIBS)

run: $(OUTPUT)
	@./$(OUTPUT)

clean:
	rm $(OUTPUT) $(OBJECTS)