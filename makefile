CXX = cl /EHsc /nologo /std:c++17

SAMPLE_SOURCE = sample/classify_wine.cpp
SAMPLE_EXE = sample/classify_wine.exe
SAMPLE_OBJ = sample/classify_wine.obj

all: clean sample

sample:
	$(CXX) /Fe:$(SAMPLE_EXE) /Fo:$(SAMPLE_OBJ) $(SAMPLE_SOURCE) /I "include" 

.PHONY: sample

clean:
	rm -f $(SAMPLE_EXE)