CFLAG = -O2 -march=native -fno-strict-aliasing -std=c++17 -masm=intel
CC = gcc 
BIN = test-amxtile 
CFILES =test-amxtile.cpp


all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean

