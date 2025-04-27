TARGET = acotsp
SRCS = main.cpp worker.cu queen.cu algorithm.cpp graph.cpp
CUDA_FLAGS = -O2 -std=c++17
LIBS = -lcuda -lcurand

all: $(TARGET)

$(TARGET): $(SRCS)
	nvcc $(CUDA_FLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
