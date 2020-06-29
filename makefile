CC = mpicxx
LIB = -I${MKLROOT}/include  -I${MKLROOT}/include   ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

all: CASPNM

CASPNM: CASPNM.cpp 
	$(CC) -o CASPNM.o CASPNM.cpp  $(LIB)
clean :
	rm *.o
