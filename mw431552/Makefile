CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
TARGET = acotsp

OBJS = main.o graph.o algorithm.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp graph.h algorithm.h
	$(CXX) $(CXXFLAGS) -c main.cpp

graph.o: graph.cpp graph.h
	$(CXX) $(CXXFLAGS) -c graph.cpp

algorithm.o: algorithm.cpp algorithm.h
	$(CXX) $(CXXFLAGS) -c algorithm.cpp

clean:
	rm -f $(TARGET) *.o
