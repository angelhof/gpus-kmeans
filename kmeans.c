#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define METHOD 2

double** create_points(int n, int dim){
    double **ps, *temp;
    int i;
    temp = (double *)calloc(n * dim, sizeof(double));
    ps = (double **)calloc(n, sizeof(double *));    
    for (i = 0 ; i < n; i++)
        ps[i] = temp + i * dim;
    if (ps == NULL || temp == NULL) {
        fprintf(stderr, "Error in allocation!\n");
        exit(-1);
    }
    return ps;
}

char** create_clusters(int k, int n){
    char **ps, *temp;
    int i;
    temp = (char *)calloc(k * n, sizeof(char));
    ps = (char **)calloc(k, sizeof(char *));    
    for (i = 0 ; i < k; i++)
        ps[i] = temp + i * n;
    if (ps == NULL || temp == NULL) {
        fprintf(stderr, "Error in allocation!\n");
        exit(-1);
    }
    return ps;
}

void delete_points(double** ps){
    free(ps);
    ps = NULL;
}

#if METHOD == 1
double* find_maxs(double** ps, int n, int dim) {
    int i,j;
    double *max;
    
    max = (double *)calloc(dim, sizeof(double));
    for (j = 0; j < dim; j++) max[j] = ps[0][j];

    for (i = 1; i < n; i++)
        for (j = 0; j < dim; j++)
            if (ps[i][j] > max[j]) max[j] = ps[i][j];
    
    return max;
}

double* find_mins(double** ps, int n, int dim) {
    int i,j;
    double *min;
    
    min = (double *)calloc(dim, sizeof(double));
    for (j = 0; j < dim; j++) min[j] = ps[0][j];

    for (i = 1; i < n; i++)
        for (j = 0; j < dim; j++)
            if (ps[i][j] < min[j]) min[j] = ps[i][j];
    
    return min;
}

double find_rand_in_range(double min, double max) {
    double diff = max-min;
    return diff * ( (double)rand() / (double)RAND_MAX ) + min;
}

double** init_centers(double *max, double *min, int k, int dim) {
    int i, j;
    double **centers;

    centers = create_points(k, dim);
    srand(time(NULL));
    for (i = 0; i < k; i++)
        for (j = 0; j < dim; j++)
            centers[i][j] = find_rand_in_range(min[j], max[j]);

    return centers;
}
#endif

#if METHOD == 2
double** init_centers(double **ps, int n, int k, int dim) {
    int i, j;
    int chosen = 0;
    double **centers;
    char *used_points;

    centers = create_points(k, dim);
    used_points = (char *)calloc(n, sizeof(char));
    srand(time(NULL));
    for (i = 0; i < k; i++) {
        do {
            chosen = rand() % n;
        } while (used_points[chosen] != 0);
        used_points[chosen] = 1;
        for (j = 0; j < dim; j++)
            centers[i][j] = ps[chosen][j];
    }
    
    return centers;
}
#endif

double distance(double* ps, double* center, int dim) {
    int i;
    double sum = 0;

    // Xreiazetai sqrt???
    for (i = 0; i < dim; i++){
        double temp = center[i] - ps[i];
        sum += temp * temp;
    }

    return sum;
}

int find_cluster(double* ps, double** centers, int n, int k, int dim) {
    int cluster = 0;
    int j;
    double dist;
    double min = distance(ps, centers[0], dim);

    for (j = 1; j < k; j++){
        dist = distance(ps, centers[j], dim);
        if (min > dist) cluster = j;
    }

    return cluster;
}

double* update_center(double** ps, char* cluster, int n, int dim) {
    int i, j, points_in_cluster = 0;
    double *new_center;

    new_center = (double *)calloc(dim + 1, sizeof(double));

    for (i = 0; i < n; i++) {
        if (cluster[i]){
            points_in_cluster++;
            for (j = 0; j < dim; j++) new_center[j] += ps[i][j];
        }
    }

    if (points_in_cluster > 0) {
        for (i = 0; i < dim; i++)
            new_center[i] /= points_in_cluster;
        new_center[dim] = 2;
    }
    else new_center[dim] = -2;
    return new_center;
}

int main() {
    // read input
    int n, k, i, j;
    int dim = 2;
    double **points;

    scanf("%d %d", &n, &k);
    points = create_points(n, dim);
    for (i = 0; i < n; i++) {
        scanf("%lf %lf", &points[i][0], &points[i][1]);
    }

    #if METHOD == 1
    // find limits
    double *max, *min;
    max = find_maxs(points, n, dim);
    min = find_mins(points, n, dim);
    
    // initiate centers
    double **centers;
    centers = init_centers(max, min, k, dim);
    #endif


    #if METHOD == 2
    // initiate centers
    double **centers;
    centers = init_centers(points, n, k, dim);
    #endif
    
    // start algorithm
    double check = 1;
    double eps = 1.0E-3;
    char **clusters;
    int *prev_clusters;
    int cl;

    clusters = create_clusters(k, n);
    prev_clusters = (int *)calloc(n, sizeof(int));

    while (check > eps) {
        // assign points
        for (i = 0; i < n; i++) {
            cl = find_cluster(points[i], centers, n, k, dim);
            int prev = prev_clusters[i];
            if (cl != prev) {
                clusters[cl][i] = 1;
                clusters[prev][i] = 0;
                prev_clusters[i] = cl;
            }
        }

        // update means
        check = 0;
        for (j = 0; j < k; j++) {
            double *new_center;
            new_center = update_center(points, clusters[j], n, dim);
            if (new_center[dim] > 0) {
                check += sqrt(distance(new_center, centers[j], dim));
                // ISWS MEMCPY ???
                for (i = 0; i < dim; i++) centers[j][i] = new_center[i];
            }
        }

    }

    // print results
    printf("Centers:\n");
    for (i = 0; i < k; i++) {
        for (j = 0; j < dim; j++)
            printf("%lf ", centers[i][j]);
        printf("\n");
    }

    // clear and exit
    delete_points(points);
    delete_points(centers);
    return 0;
}