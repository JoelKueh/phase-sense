
OPT_LEVEL= -O2

OUT_DIR=out
OBJ_DIR=obj
BIN_DIR=bin
SRC_DIR=src

INCLUDE_DIRS=-I./src -I./inc
LD_FLAGS=-lGL -lglfw -lm

.PHONY: default clean_all clean debug
default: $(BIN_DIR)/dtwin

debug: OPT_LEVEL= 
debug: CFLAGS += -g
debug: default

CFLAGS = $(INCLUDE_DIRS) $(LD_FLAGS) $(OPT_LEVEL) -std=c23

SHADER_SRCS = $(SRC_DIR)/shaders/inst_vert.glsl $(SRC_DIR)/shaders/inst_geom.glsl \
	$(SRC_DIR)/shaders/inst_frag.glsl $(SRC_DIR)/shaders/quad_vert.glsl \
	$(SRC_DIR)/shaders/gaus_frag.glsl

DTWIN_SRCS = $(SRC_DIR)/dtwin.c $(SRC_DIR)/render.c $(SRC_DIR)/nbody.c $(SRC_DIR)/glad.c
DTWIN_OBJS = $(DTWIN_SRCS:src/%.c=obj/%.o)
DTWIN_DEPS = $(DTWIN_OBJS:%.o=%.d)
-include $(DTWIN_DEPS)

obj/shaders.o: src/shaders.c $(SHADER_SRCS)
	@mkdir -p $(dir $@)
	$(CC) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< $(CFLAGS) 

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< $(CFLAGS) 

$(BIN_DIR)/dtwin: obj/dtwin.o $(DTWIN_OBJS) obj/shaders.o
	@mkdir -p $(dir $@)
	@mkdir -p $(OUT_DIR)
	$(CC) -o $@ obj/shaders.o $(DTWIN_OBJS) $(CFLAGS)

clean_all: clean
	rm -rf ./$(OUT_DIR)/*
	
clean:
	rm -rf ./$(OBJ_DIR)/* ./$(BIN_DIR)/*
