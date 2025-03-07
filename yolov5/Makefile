CC = gcc
CFLAGS = -O3
CFLAGS += -Wall
#CFLAGS += -g

main:main.c utils.h text2img.h yolov5.h
	${CC} $(CFLAGS) -o $@ main.c -lrknnrt -lm

clean:
	rm -rf main results
