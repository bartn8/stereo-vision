CC=g++
DEST_ARCH=#-m32 -L/usr/lib32
OPENMP= -fopenmp
#OPENCV= -lopencv_core -lopencv_highgui
OPENCV= $(shell pkg-config --libs opencv4)
#CFLAGS=-I. -msse4.1 -msse4.2 -O3 -ffast-math  -march=core2 $(DEST_ARCH) $(OPENMP) -Wno-write-strings
CFLAGS=-I. -msse4.1 -msse4.2 -O3 -ffast-math  -march=native $(DEST_ARCH) $(OPENMP) $(shell pkg-config --cflags opencv4) -Wno-write-strings
LDFLAGS= $(DEST_ARCH) $(OPENMP) $(OPENCV)
ODIR=./
DEPS = StereoBMHelper.h
_OBJ = rSGMCmd.o FastFilters.o StereoBMHelper.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

rSGMCmd: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ rSGMCmd
