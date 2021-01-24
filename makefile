CXX = cl
CXXFLAGS = /EHsc /nologo /std:c++17 /O2

MNIST_SOURCE = mnist/mnist.cpp
MNIST_EXE = mnist/mnist.exe
MNIST_OBJ = mnist/mnist.obj

all: clean mnist

mnist:
	$(CXX) $(CXXFLAGS) /Fe:$(MNIST_EXE) /Fo:$(MNIST_OBJ) $(MNIST_SOURCE) /I "include"

.PHONY: mnist

clean:
	rm -f $(MNIST_EXE) $(MNIST_OBJ)