# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=out
OBJECTS=BitPos.o H264parser.o

#headers
INC=-I/usr/local/cuda-6.5/include

#libraries
LIBS=-lnvcuvid -lcuda -lcudart
LIB_PATHS=-L/usr/lib/nvidia-340

#targets
all: $(MAIN)
	$(CUDA) $(MAIN) $(OBJECTS) -o $(OUTPUT) $(INC) $(LIB_PATHS) $(LIBS)

classes: bitpos parser

bitpos: BitPos.cpp
	$(STD) BitPos.cpp -c $(INC) $(LIB_PATHS) $(LIBS)

parser: H264parser.cpp
	$(STD) H264parser.cpp -c $(INC) $(LIB_PATHS) $(LIBS)

run: $(OUTPUT)
	@./$(OUTPUT)

clean:
	rm $(OUTPUT) $(OBJECTS)