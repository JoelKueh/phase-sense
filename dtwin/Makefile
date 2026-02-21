
OPT_LEVEL= -O2

OBJ_DIR=obj
BIN_DIR=bin
SRC_DIR=src

INCLUDE_DIRS= -I/usr/local/cuda/include -I./include/
LD_FLAGS = -lGL -ldl
CUDA_LD_FLAGS = -L/usr/local/cuda/lib64 -lcudart -L./bin -lnbody_cu

.PHONY: default clean debug
default: $(BIN_DIR)/perft

debug: OPT_LEVEL= 
debug: CFLAGS += -g
debug: default

CFLAGS = $(INCLUDE_DIRS) $(LD_FLAGS) $(OPT_LEVEL) -std=c++11

DTWIN_SRCS = dtwin.cpp
DTWIN_OBJS = $(DTWIN_SRCS:src/%.cpp=obj/%.o)
DTWIN_DEPS = $(DTWIN_OBJS:%.o=%.d)
-include $(DTWIN_DEPS)

obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	g++ $(CFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

libnbody_cu.a: nbody_cu.cu
	nvcc -rdc=true -c -o ./$(OBJ_DIR)/temp.o nbody_cu.cu -O2 -gencode=arch=compute_86,code=\"sm_86\"
	nvcc -dlink -o ./$(OBJ_DIR)/nbody_cu.o ./$(OBJ_DIR)/temp.o -lcudart -gencode=arch=compute_86,code=\"sm_86\"
	ar rcs ./$(BIN_DIR)/libnbody_cu.a ./$(OBJ_DIR)/nbody_cu.o ./$(OBJ_DIR)/temp.o
	ranlib ./$(BIN_DIR)/libnbody_cu.a

.PHONY: $(BIN_DIR)/dtwin
$(BIN_DIR)/dtwin: obj/debug.o $(DTWIN_OBJS)
	@mkdir -p $(dir $@)
	g++ $(CFLAGS) $(CUDA_LD_FLAGS) -o $@ glad/glad.c $^

clean:
	rm -rf ./$(OBJ_DIR)/* ./$(BIN_DIR)/*
