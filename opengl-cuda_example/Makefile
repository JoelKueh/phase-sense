report/index.html: report/report.typ
	typst compile --features html --format html report/report.typ report/index.html

#nbody_cu.so: nbody_cu.cu
#	nvcc --shared -o nbody_cu.so nbody_cu.cu -O2 -I/opt/cuda/include -gencode=arch=compute_86,code=\"sm_86\"

nbody: nbody_app.cpp nbody_cu.h
	nvcc nbody_app.cpp nbody_cu.cu glad/glad.c -lSDL3 -lGL -ldl -I/opt/cuda/include -I/usr/include/SDL3 -O2 -o nbody

macle119_final.tar.gz:
	tar -czf macle119_final.tar.gz Makefile nbody_app.cpp glad glm nbody_cu.cu nbody_cu.h

