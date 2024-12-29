# Makefile
CC = gcc
CFLAGS = -fopenmp -Wall -O2
TARGET = matrix_mult
SRC = matrix_mult.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)