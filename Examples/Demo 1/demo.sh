gcc -o singleThread singleThread.c
echo "Az egy szalon valo futas ideje: "
time ./singleThread polygons.txt results_s.csv

nvcc -arch=sm_50 -o gpu gpu.cu
echo "Az GPU futas ideje: "
time ./gpu polygons.txt results_g.csv