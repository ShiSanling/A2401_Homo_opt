CC = g++
CFLAGS = -I/dssg/home/acct-mezyx/mezyx-jzk/Software/petsc/include/ -I/dssg/home/acct-mezyx/mezyx-jzk/Software/petsc/arch-linux-c-debug/include
LDFLAGS = -L/dssg/home/acct-mezyx/mezyx-jzk/Software/petsc/arch-linux-c-debug/lib
LIBS = -lpetsc -lmpi

all: ksp

ksp: ksp.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o ksp ksp.cpp $(LIBS)

clean:
	rm -f del
