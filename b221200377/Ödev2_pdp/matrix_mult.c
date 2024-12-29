#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Matris belleği ayırma fonksiyonu
double **allocate_matrix(int rows, int cols) {
    double *data = (double *)calloc(rows * cols, sizeof(double));
    double **matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = &data[i * cols];
    }
    return matrix;
}

// Matris belleğini serbest bırakma fonksiyonu
void free_matrix(double **matrix) {
    free(matrix[0]); // Veri blokunu serbest bırak
    free(matrix);    // Satır göstergelerini serbest bırak
}

// Bir dosyadan matrisi okuma fonksiyonu
void read_matrix(const char *filename, int *rows, int *cols, double ***matrix) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Dosya açma hatası");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Dosya format hatası: %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *matrix = allocate_matrix(*rows, *cols);

    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%lf", &((*matrix)[i][j])) != 1) {
                fprintf(stderr, "Matris verisi okunamadı: %s\n", filename);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

// Matrisleri seri olarak çarpan fonksiyon
void multiply_serial(double **A, double **B, double ***C, int rowsA, int colsA, int colsB) {
    *C = allocate_matrix(rowsA, colsB);

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                (*C)[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrisleri OpenMP kullanarak paralel olarak çarpan fonksiyon
void multiply_parallel(double **A, double **B, double ***C, int rowsA, int colsA, int colsB) {
    *C = allocate_matrix(rowsA, colsB);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                (*C)[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrisi bir dosyaya yazan fonksiyon
void write_matrix(const char *filename, double **matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Dosya açma hatası");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.2lf ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Kullanım: %s <matrix_a.txt> <matrix_b.txt> <sonuclar.txt>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int rowsA, colsA, rowsB, colsB;
    double **A, **B, **C_serial, **C_parallel;

    read_matrix(argv[1], &rowsA, &colsA, &A);
    read_matrix(argv[2], &rowsB, &colsB, &B);

    if (colsA != rowsB) {
        fprintf(stderr, "Matris boyutları çarpım için uyumlu değil.\n");
        free_matrix(A);
        free_matrix(B);
        return EXIT_FAILURE;
    }

    clock_t start, end;

    // Seri çarpım
    start = clock();
    multiply_serial(A, B, &C_serial, rowsA, colsA, colsB);
    end = clock();
    double serial_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Paralel çarpım
    start = clock();
    multiply_parallel(A, B, &C_parallel, rowsA, colsA, colsB);
    end = clock();
    double parallel_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Sonuçları dosyaya yazma
    write_matrix(argv[3], C_parallel, rowsA, colsB);

    // Zaman karşılaştırmasını yazdırma
    printf("Seri zaman: %.6f saniye\n", serial_time);
    printf("Paralel zaman: %.6f saniye\n", parallel_time);

    // Belleği serbest bırakma
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_serial);
    free_matrix(C_parallel);

    return EXIT_SUCCESS;
}
