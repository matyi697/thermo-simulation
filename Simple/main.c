#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 100           // a szimulacio x merete
#define HEIGHT 100          // a szimulacio y merete
#define DX 1.0              // x tengely felbontasa
#define DY 1.0              // y tengely felbontasa
#define K 0.1               // hovezetesi egyutthato
#define MAX_POLYGONS 100    // max polygon szam
#define ITERATIONS 5000     // szimulacio itercioi (hanyszor fusson az automata)

typedef struct {
    double vertices[MAX_POLYGONS][2];   // polygon pontjai (x,y)
    int num_vertices;   
    double power;                       // A polygon disszipacios teljesitmenye (W)
} Polygon;

int read_polygons(char *file_path, Polygon polygons[]) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Bemeneti file hiba! (polygonok file)\n");
        return -1;
    }

    char line[256];
    int num_polygons = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'p') {               // polygonok beolvasasa (p-vel kezdodo sorok)
            double x, y, power;
            sscanf(line + 1, "%lf %lf %lf", &x, &y, &power);
            Polygon *poly = &polygons[num_polygons];     //hogy ne kelljen minden polygon[] hasznalni igy szebb   
            poly->vertices[poly->num_vertices][0] = x;
            poly->vertices[poly->num_vertices][1] = y;
            poly->power = power;
            poly->num_vertices++;
        } else {
            if (polygons[num_polygons].num_vertices > 0) {      //ha nem p sor akkor uj polygon
                num_polygons++;
            }
        }
    }

    fclose(file);
    return num_polygons;
}

/*ez a fuggveny egy okos modon nezi meg hogy egy pont benne van-e egy polygonban (halado pont algoritmus)*/
int is_point_in_polygon(double x, double y, Polygon *polygon) {    
    int inside = 0;
    int j = polygon->num_vertices - 1;

    for (int i = 0; i < polygon->num_vertices; i++) {
        if ((polygon->vertices[i][1] > y) != (polygon->vertices[j][1] > y) &&   /*ez a resz ellenőrzi, hogy a pont a polygon melyik oldalan van*/
            (x < (polygon->vertices[j][0] - polygon->vertices[i][0]) * (y - polygon->vertices[i][1]) /  /*ezek a polygon oldalainak egyenletei alapjan nezi metszi-e*/
            (polygon->vertices[j][1] - polygon->vertices[i][1]) + polygon->vertices[i][0])) {
            inside = !inside;   /*minden metszes valt (paros = nincs benne, paratlan = benne)*/
        }
        j = i;
    }
    return inside;
}

/*a hosugarzo blokkot minden iteracioban frissiteni kell az eredeti homersekletre*/
void update_temperature(double temperature[HEIGHT][WIDTH], Polygon polygons[], int num_polygons) {
    for (int i = 0; i < num_polygons; i++) {
        double polygon_power = polygons[i].power;

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (is_point_in_polygon(x, y, &polygons[i])) {
                    temperature[y][x] = polygon_power;
                }
            }
        }
    }
}

void simulate_heat_conduction(double temperature[HEIGHT][WIDTH], Polygon polygons[], int num_polygons) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double new_temperature[HEIGHT][WIDTH] = {0};

        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                /*a diszkret hoterjedesi modell matematikai alakja implementalva a gridre*/
                new_temperature[y][x] = temperature[y][x] + K * (
                    (temperature[y + 1][x] - 2 * temperature[y][x] + temperature[y - 1][x]) / (DY * DY) +
                    (temperature[y][x + 1] - 2 * temperature[y][x] + temperature[y][x - 1]) / (DX * DX)
                );
            }
        }

        update_temperature(new_temperature, polygons, num_polygons);

        memcpy(temperature, new_temperature, sizeof(double) * HEIGHT * WIDTH);  //eredeti tommbe valo masolas (frissitese a reginek)
    }
}

void write_results(const char *file_path, double temperature[HEIGHT][WIDTH]) {
    FILE *output_file = fopen(file_path, "w");
    if (output_file == NULL) {
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
        printf("Helytelen argumentumok, helyes hasznalat:\n\t -%s [polygon file].txt [kimenet].csv [iteraciok / szimulacio]\n", argv[0]);
        return 1;
    }

    double temperature[HEIGHT][WIDTH] = {0};
    Polygon polygons[MAX_POLYGONS] = {0};
    int num_polygons = read_polygons(argv[1], polygons);
    if (num_polygons < 0) return 1;

    update_temperature(temperature, polygons, num_polygons);

    simulate_heat_conduction(temperature, polygons, num_polygons);

    write_results(argv[2], temperature);
    
    printf("Kesz!\n");
    return 0;
}