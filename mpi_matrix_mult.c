 /******************************************************************************
* FILE: mpi_matrix_mult.c
* DESCRIPTION:  
* AUTHOR: Jessica de Leeuw. Adapted from Blaise Barney
******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define MASTER 0        /* task_rank of first task */
#define FROM_MASTER 1   /* setting a message type */
#define FROM_WORKER 2   /* setting a message type */

int M;       /* number of rows in matrix A */
int N;       /* number of columns in matrix A */
int scheme;

int main (int argc, char *argv[]) {

    /* User input determines number of rows and columns*/
    N = atoi(argv[1]);
    M = atoi(argv[2]);

    /* 0 --> Row Major, 1 --> Column Major */
    scheme = atoi(argv[3]);

    int	num_tasks,             /* number of tasks in partition */
        task_rank,             /* a task identifier */
        num_workers,           /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        msg_type,              /* message type */
        rows,                  /* rows of matrix A sent to each worker */
        av_row, extra, offset; /* used to determine rows sent to each worker */
        // rc;          

    double	a[N][M], /* matrix A to be multiplied */
            b[M][N], /* matrix B to be multiplied */
            c[N][N], /* result matrix C */
            f[M * N];
    
    double  t1,           /* start time of process */ 
            t2,           /* finish time of process */
            process_time, /* time for each process */
            total_time;   /* total time taken */

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    total_time = 0; 

    if (num_tasks < 2 ) {
        printf("Need at least two MPI tasks. \n");
        /* Proper error code to return instead of -1?? */
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(1);
    }
    num_workers = num_tasks - 1;


    /**************************** master ************************************/
    if (task_rank == MASTER) {

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                a[i][j]= (i + 1) + (j + 1);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                b[i][j]= (i + 1) * (j + 1);
        /*
        // Fill matrices with randomly generated numbers 
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++)
                a[i][j] = drand48();
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
            b[i][j]= drand48();
        }
        */

        if (scheme == 0) { // calculation when matrix is row-major

            /* Send matrix data to the worker tasks */
            av_row = M / num_workers; // rows of A that each worker will operate on
            extra = M % num_workers;  // any extra miscellaneous rows are sent to one of the workers
            offset = 0;
            msg_type = FROM_MASTER;

            for (dest = 1; dest <= num_workers; dest++) {
                rows = (dest <= extra) ? av_row + 1 : av_row;   	
                printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
                MPI_Send(&offset, 1, MPI_INT, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&a[offset][0], rows * N, MPI_DOUBLE, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&b, N * M, MPI_DOUBLE, dest, msg_type, MPI_COMM_WORLD);
                offset += rows;
            }
        }
        else { /* Change this to be COLUMN MAJOR */
            /* Send matrix data to the worker tasks */
            av_row = M / num_workers; // rows of A that each worker will operate on
            extra = M % num_workers;  // any extra miscellaneous rows are sent to one of the workers
            offset = 0;
            msg_type = FROM_MASTER;

            /* flatten matrix B to computer in column major form */
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    f[i * N + j] = b[i][j];  
                }
            }

            for (dest = 1; dest <= num_workers; dest++) {
                rows = (dest <= extra) ? av_row + 1 : av_row;   	
                printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
                MPI_Send(&offset, 1, MPI_INT, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&a[offset][0], rows * N, MPI_DOUBLE, dest, msg_type, MPI_COMM_WORLD);
                MPI_Send(&f, N * M, MPI_DOUBLE, dest, msg_type, MPI_COMM_WORLD);
                offset += rows;
            }
        }
    }

    /**************************** worker task ************************************/
    if (task_rank > MASTER) {

        if (scheme == 0) {
            msg_type = FROM_MASTER;
            MPI_Recv(&offset, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&a, rows * N, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&b, N * M, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD, &status);

            t1 = MPI_Wtime();
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < rows; i++) {
                    c[i][k] = 0.0;
                    for (int j = 0; j < M; j++)
                        c[i][k] = c[i][k] + a[i][j] * b[j][k]; 
                }
            }
            t2 = MPI_Wtime();

            process_time = t2 - t1;
        }
        else {
            msg_type = FROM_MASTER;
            MPI_Recv(&offset, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&a, rows * N, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&f, N * M, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD, &status);

            t1 = MPI_Wtime();
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < rows; i++) {
                    c[i][k] = 0.0;
                    for (int j = 0; j < M; j++)
                        c[i][k] = c[i][k] + a[i][j] * f[j * N + k];  
                }
            }
            t2 = MPI_Wtime();

            process_time = t2 - t1;
        }

        msg_type = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, msg_type, MPI_COMM_WORLD);
        MPI_Send(&c, rows * N, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD);
        MPI_Send(&process_time, __SIZEOF_DOUBLE__, MPI_DOUBLE, MASTER, msg_type, MPI_COMM_WORLD);
    }

    if (task_rank == MASTER) {
        /* Receive results from worker tasks */
        msg_type = FROM_WORKER;
        for (int i = 1; i <= num_workers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * N, MPI_DOUBLE, source, msg_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&process_time, __SIZEOF_DOUBLE__, MPI_DOUBLE, source, msg_type, MPI_COMM_WORLD, &status);
            total_time += process_time;
        }
        printf("Time taken: %lf\n", total_time);
        
        /*
        printf("******************************************************\n");
        printf("Result Matrix:\n");
        for (int i = 0; i < M; i++) {
            printf("\n"); 
            for (int j = 0; j < N; j++) 
            printf("%6.2f   ", c[i][j]);
        }
        printf("\n******************************************************\n"); 
        */
    }
    MPI_Finalize();
}