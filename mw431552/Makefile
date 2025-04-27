NVCC = nvcc
CXXFLAGS = -O2 -std=c++17
LDFLAGS = -lcuda -lcurand

TARGET = acotsp
SRCS = main.cpp worker.cu algorithm.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o