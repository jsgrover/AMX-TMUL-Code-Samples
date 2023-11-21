CFLAG = -fopenmp  -march=native -std=c++23  -g -lstdc++ -Wall
CC = gcc 
BIN = test-amxtile 
CFILES =test-amxtile.cpp


all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean

