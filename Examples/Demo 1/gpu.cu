#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define WIDTH 100 
#define HEIGHT 100    
#define DX 1.0              
#define DY 1.0              
#define K 0.1               
#define MAX_POLYGONS 100    
#define ITERATIONS 5000     

typedef struct {
    double vertices[MAX_POLYGONS][2];   
    int num_vertices;   
    double power;                       
} Polygon;

__host__ int read_polygons(const char *file_path, Polygon polygons[]) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        perror("Bemeneti file hiba! (polygonok file)\n");
        return -1;
    }

    char line[256];
    int num_polygons = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'p') {              
            double x, y, power;
            sscanf(line + 1, "%lf %lf %lf", &x, &y, &power);
            Polygon *poly = &polygons[num_polygons];
            poly->vertices[poly->num_vertices][0] = x;
            poly->vertices[poly->num_vertices][1] = y;
            poly->power = power;
            poly->num_vertices++;
        } else {
            if (polygons[num_polygons].num_vertices > 0) {
                num_polygons++;
            }
        }
    }

    fclose(file);
    return num_polygons;
}

__device__ int is_point_in_polygon(double x, double y, Polygon *polygon) {
    int inside = 0;
    int j = polygon->num_vertices - 1;

    for (int i = 0; i < polygon->num_vertices; i++) {
        if ((polygon->vertices[i][1] > y) != (polygon->vertices[j][1] > y) &&
            (x < (polygon->vertices[j][0] - polygon->vertices[i][0]) * (y - polygon->vertices[i][1]) / 
            (polygon->vertices[j][1] - polygon->vertices[i][1]) + polygon->vertices[i][0])) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

__global__ void update_temperature_kernel(double *temperature, Polygon *polygons, int num_polygons) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return; 
    //ha nagyobb a szal index mint a tomb max index (5 ora debug utan jottem ra)

    double polygon_power = 0.0;
    for (int i = 0; i < num_polygons; i++) {
        if (is_point_in_polygon((double)x, (double)y, &polygons[i])) {
            polygon_power = polygons[i].power;
            temperature[y * WIDTH + x] = polygon_power;
        }
    }
}

__global__ void simulate_heat_conduction_kernel(double *temperature, double *new_temperature) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT || x == 0 || y == 0 || x == WIDTH - 1 || y == HEIGHT - 1) return;
    
    /*ezt meg nem ertem teljesen*/
    new_temperature[y * WIDTH + x] = temperature[y * WIDTH + x] + K * (
        (temperature[(y + 1) * WIDTH + x] - 2 * temperature[y * WIDTH + x] + temperature[(y - 1) * WIDTH + x]) / (DY * DY) +
        (temperature[y * WIDTH + (x + 1)] - 2 * temperature[y * WIDTH + x] + temperature[y * WIDTH + (x - 1)]) / (DX * DX)
    );
}

void simulate_heat_conduction(double *temperature, Polygon *polygons, int num_polygons) {
    double *d_temperature, *d_new_temperature;
    Polygon *d_polygons;

    cudaMalloc((void**)&d_temperature, sizeof(double) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_new_temperature, sizeof(double) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_polygons, sizeof(Polygon) * num_polygons);

    cudaMemcpy(d_temperature, temperature, sizeof(double) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_polygons, polygons, sizeof(Polygon) * num_polygons, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        update_temperature_kernel<<<numBlocks, threadsPerBlock>>>(d_temperature, d_polygons, num_polygons);
        simulate_heat_conduction_kernel<<<numBlocks, threadsPerBlock>>>(d_temperature, d_new_temperature);
        
        // Swap the temperature arrays
        double *temp = d_temperature;
        d_temperature = d_new_temperature;
        d_new_temperature = temp;
    }

    cudaMemcpy(temperature, d_temperature, sizeof(double) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    cudaFree(d_temperature);
    cudaFree(d_new_temperature);
    cudaFree(d_polygons);
}

void write_results(const char *file_path, double temperature[HEIGHT][WIDTH]) {
    FILE *output_file = fopen(file_path, "w");
    if (!output_file) {
        perror("Hiba az eredmenyek fileba irasakor!\n");
        return;
    }

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            fprintf(output_file, "%.2f,", temperature[y][x]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Helytelen argumentumok, helyes hasznalat:\n\t -%s [polygon file].txt [kimenet].csv\n", argv[0]);
        return 1;
    }

    double temperature[HEIGHT][WIDTH] = {0};
    Polygon polygons[MAX_POLYGONS] = {0};
    
    int num_polygons = read_polygons(argv[1], polygons);
    if (num_polygons < 0) return 1; 

    simulate_heat_conduction(&temperature[0][0], polygons, num_polygons);

    write_results(argv[2], temperature);
    
    printf("Kesz!\n");
    return 0;
}
