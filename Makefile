CC = g++
CFLAGS = -I$(PETSC_DIR)/include/ -I$(PETSC_DIR)/arch-linux-c-debug/include
LDFLAGS = -L$(PETSC_DIR)/arch-linux-c-debug/lib
LIBS = -lpetsc -lmpi


all: ksp

ksp: ksp.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o ksp ksp.cpp $(LIBS)

refer: refer.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o refer refer.cpp $(LIBS)
clean:
	rm -f del
