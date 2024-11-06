#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define WIDTH 400           //  rács szélessége
#define HEIGHT 400          //  rács magassága
#define DX 1.0              //  hővezetési szimuláció x febontása
#define DY 1.0              //  hővezetési szimuláció y febontása
#define K 0.1               //  hővezetési együttható
#define MAX_POLYGONS 100    //  maximális polygon tömb méret
#define ITERATIONS 5000     //  szimuláció iterációi

/*
Ebben a structban tároljuk a polygonokat
- a vertices a lapok
- a power a hőteljesítmény amit leadnak
Ebből a structbol ha több van akkor meg lehet valósítani több oldalú es több külön álló alakzatot is
*/
typedef struct {
    double vertices[MAX_POLYGONS][2];   
    int num_vertices;   
    double power;                       
} Polygon;

/*
Ez egy host (cpu) függvény ami beolvassa a polygon adatokat a txt fileból
*/
__host__ int read_polygons(const char *file_path, Polygon polygons[]) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        perror("Bemeneti file hiba! (polygonok file)\n");
        return -1;
    }

    char line[256];
    int num_polygons = 0;
    
    //beolvassuk az adatokat sorrol sorra a structba
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

/*
Ez egy device függvény ami a gpu-n fut és a gpu hívja meg.
Az utazó pont algoritmussal meg tudja mondani hogy egy pont benne van-e egy polygonban
az algoritmus forrása: https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
*/
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

/*
Ez egy global függvény ami a gpu-n fut és a cpu hívja meg, minden iteráció elején frissíteni kell a hősugárzó pontokat mivel a
hőterjedés alatt "lehültek", a tömb elemeit egyesével egymástól függetlenül dolgozza fel
*/
__global__ void update_temperature_kernel(double *temperature, Polygon *polygons, int num_polygons) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  /*Megcsináljuk az egyedi szál-indexeket x-re és y-ra*/
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return; /*A szálak indexei ha meghaladják a tömb indexelést visszatér*/

    double polygon_power = 0.0;
    for (int i = 0; i < num_polygons; i++) {
        if (is_point_in_polygon((double)x, (double)y, &polygons[i])) {
            polygon_power = polygons[i].power;
            temperature[y * WIDTH + x] = polygon_power;
        }
    }
}

/*
Ez egy global függvény, ez a diszkrét hővezetési matematikai modell implementáció, minden mezőt egyszerre dolgoz fel a gpu.
A frissített hőmérséklet adatokat az argumentum vektoron kapott new_temperature-ben tárolja.
A matematikai egyenlet megértéséhez ezt a videót: https://www.youtube.com/watch?v=S8mopZlhRHE
és ezt a wikipédia oldalt használtam fel: https://en.wikipedia.org/wiki/Heat_equation
*/
__global__ void simulate_heat_conduction_kernel(double *temperature, double *new_temperature) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  //Megcsináljuk az egyedi szál-indexeket x-re és y-ra
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    //Túlindexelés védelem
    if (x >= WIDTH || y >= HEIGHT || x == 0 || y == 0 || x == WIDTH - 1 || y == HEIGHT - 1) return;
    
    //A diszkrét hővezetés modell matematikai alakja
    new_temperature[y * WIDTH + x] = temperature[y * WIDTH + x] + K * (
        (temperature[(y + 1) * WIDTH + x] - 2 * temperature[y * WIDTH + x] + temperature[(y - 1) * WIDTH + x]) / (DY * DY) +
        (temperature[y * WIDTH + (x + 1)] - 2 * temperature[y * WIDTH + x] + temperature[y * WIDTH + (x - 1)]) / (DX * DX)
    );
}

/*
Ez a keret függvény másolja át a változókat a videókártya memóriába (azokat a változókat amik a gpu-n vannak foglalva
egy d_-al (d=device) jeleztem amit a változó neve követ), a másolás utan futtatja a szimulációt az adathalmazon és
frissíti a hősugárzó pontokat minden iteráció végén.
*/
void simulate_heat_conduction(double *temperature, Polygon *polygons, int num_polygons) {
    double *d_temperature, *d_new_temperature;
    Polygon *d_polygons;
    
    /*memóriát foglalunk a GPU-n, a jelenlegi hőmérséklet tömbnek,
    a kimeneti tömbnek (eredmény) és a polygon tömbnek*/
    cudaMalloc((void**)&d_temperature, sizeof(double) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_new_temperature, sizeof(double) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_polygons, sizeof(Polygon) * num_polygons);
    
    /*átmásoljuk az adatokat*/
    cudaMemcpy(d_temperature, temperature, sizeof(double) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_polygons, polygons, sizeof(Polygon) * num_polygons, cudaMemcpyHostToDevice);
    
    /*a gpu mag és szálak konfigurálása, egy blokkban 16*16 szál van*/
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    /*futtatjuk a modellt iter darab iterációra*/
    for (int iter = 0; iter < ITERATIONS; iter++) {
        
        /*ezek a gpu-n való global függvényhívások*/
        update_temperature_kernel<<<numBlocks, threadsPerBlock>>>(d_temperature, d_polygons, num_polygons);
        simulate_heat_conduction_kernel<<<numBlocks, threadsPerBlock>>>(d_temperature, d_new_temperature);
        
        /*a régi adatokat felül írjuk a new_temperature tömbbel, ez lesz a következő iterációban az adat*/
        double *temp = d_temperature;
        d_temperature = d_new_temperature;
        d_new_temperature = temp;
    }
    /*kimásoljuk az utolsó eredményt*/
    cudaMemcpy(temperature, d_temperature, sizeof(double) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    /*felszabadítjuk a videókártya memóriát*/
    cudaFree(d_temperature);
    cudaFree(d_new_temperature);
    cudaFree(d_polygons);
}
/*
Ez a függvény egy csv fileba menti a hőmérséklet adatokat
*/
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

/*
A program 2 parancssori argumentummal dolgozik:
-argv[1] : a polygon file neve
-argv[2] : a kimeneti csv file neve
*/
int main(int argc, char* argv[]) {
    if (argc != 3) {
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
