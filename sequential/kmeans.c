#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define METHOD 2

// #define DEBUG

#ifdef DEBUG
#define DPRINTF(fmt, args...) \
do { \
    printf("%s, line %u: " fmt "\r\n", __FUNCTION__, __LINE__ , ##args); \
    fflush(stdout); \
} while (0)
#else   
#define DPRINTF(fmt, args...)   do{}while(0)
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



// #if METHOD == 2
// double** init_centers(double **ps, int n, int k, int dim) {
//     int i, j;
//     int chosen = 0;
//     double **centers;
//     char *used_points;

//     centers = create_points(k, dim);
//     used_points = (char *)calloc(n, sizeof(char));
//     srand(time(NULL));
//     for (i = 0; i < k; i++) {
//         do {
//             chosen = rand() % n;
//         } while (used_points[chosen] != 0);
//         used_points[chosen] = 1;
//         for (j = 0; j < dim; j++)
//             centers[i][j] = ps[chosen][j];
//     }
    
//     return centers;
// }
// #endif


double** init_centers_kpp(double **ps, int n, int k, int dim){
    int i,j;
    int curr_k = 0;
    int first_i;
    int max, max_i;
    double distances_from_centers[n];
    double **centers = create_points(k,dim);
    double temp_distances[n];

    // Initialize with max double
    for (int i = 0; i < n; i++) distances_from_centers[i] = DBL_MAX;

    srand(time(NULL));

    // Choose a first point
    first_i = rand() % n;
    DPRINTF("First random index: %d", first_i);

    memcpy(centers[curr_k], ps[first_i], dim * sizeof(double));
    DPRINTF("Point 1: (%f, %f)", ps[first_i][0], ps[first_i][1]);
    DPRINTF("Center 1: (%f, %f)", centers[curr_k][0], centers[curr_k][1]);

    while(curr_k < k-1) {
        max = -1;
        max_i = -1;
        for(i=0;i<n;i++){
            DPRINTF("New distance: %f and old min distance: %f", distance(ps[i], centers[curr_k], dim), distances_from_centers[i]);
            temp_distances[i] = MIN(distance(ps[i], centers[curr_k], dim), distances_from_centers[i]);    
            if(temp_distances[i] > max){
                max = temp_distances[i];
                max_i = i;
            }
        }
 

        memcpy(distances_from_centers, temp_distances, n * sizeof(double));
        memcpy(centers[++curr_k], ps[max_i], dim * sizeof(double));
    }   
    return centers;
}



int find_cluster(double* ps, double** centers, int n, int k, int dim) {
    int cluster = 0;
    int j;
    double dist;
    double min = distance(ps, centers[0], dim);

    for (j = 1; j < k; j++){
        dist = distance(ps, centers[j], dim);
        if (min > dist){
            min = dist;
            cluster = j;
        }
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
    int dim = 3;
    double **points;

    scanf("%d %d", &n, &k);
    points = create_points(n, dim);
    for (i = 0; i < n; i++) {
        for (j = 0; j < dim; j++)
            scanf("%lf", &points[i][j]);
    }

    
    // #if METHOD == 2
    // // initiate centers
    // double **centers;
    // centers = init_centers(points, n, k, dim);
    // #endif
    
    double **centers;
    centers = init_centers_kpp(points, n, k, dim);

    // start algorithm
    double check = 1;
    double eps = 1.0E-6;
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
