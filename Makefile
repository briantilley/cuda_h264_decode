# compilers
CUDA=nvcc
STD=g++

#filenames
MAIN=main.cu
OUTPUT=a.out

#targets
all: main.cu
	nvcc $(MAIN) -o $(OUTPUT)

run: $(OUTPUT)
	./$(OUTPUT)
