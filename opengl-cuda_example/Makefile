nbody: nbody_app.cpp libnbody_cu.a
	g++ nbody_app.cpp glad/glad.c -L. -lnbody_cu -I. -lSDL3 -lGL -ldl -I/usr/include/SDL3 -L/opt/cuda/lib64 -lcudart -O2 -o nbody

#gcc nbody_app.cpp glad/glad.c -L. -l:nbody_cu.a -lSDL3 -lGL -ldl -I/opt/cuda/include -I/usr/include/SDL3 -O2 -o nbody

libnbody_cu.a: nbody_cu.cu
	nvcc -rdc=true -c -o temp.o nbody_cu.cu -O2 -gencode=arch=compute_86,code=\"sm_86\"
	nvcc -dlink -o nbody_cu.o temp.o -lcudart -gencode=arch=compute_86,code=\"sm_86\"
	ar rcs libnbody_cu.a nbody_cu.o temp.o
	ranlib libnbody_cu.a

clean:
	rm nbody
	rm *.o
	rm *.a
