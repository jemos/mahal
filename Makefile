mahal: mahal.c
	gcc -g -lc -O3 -mfpmath=sse -std=c99 -pedantic -Wall mahal.c -o mahal

debug: mahal.c
	gcc -g -lc -std=c99 -pedantic -Wall mahal.c -o mahal
