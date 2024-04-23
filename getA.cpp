static char help[] = "linear solve";
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>

Mat Get_A(char* Ain)
{
    Mat A;
    PetscErrorCode ierr;
    PetscInt row,col,rstart,rend;
    int m, n,nz, dummy, bindex;
    PetscScalar val;
    FILE *Afile;
    PetscMPIInt size, rank;

    PetscInitialize(NULL, NULL, NULL, NULL);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    

    Afile = fopen(Ain, "r");
    fscanf(Afile, "%d %d %d\n", &m, &n, &nz);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);
    
    ierr = MatSetFromOptions(A);
    
    MatSetUp(A);
    ierr = MatGetOwnershipRange(A, &rstart, &rend);
    
    // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart,rend-1);
    for (int i = 0; i < nz; i++)
    {
        fscanf(Afile, "%d %d %le\n", &row, &col, (double *)&val);
        if (row >= rstart && row < rend)
            ierr = MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES);
        
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    fflush(stdout);
    fclose(Afile);
    PetscFinalize();
    return A;
}


int main(int argc, char **args)
{
    char Ain[PETSC_MAX_PATH_LEN];
    PetscBool flg_A;
    PetscErrorCode ierr;
    Mat A;
    ierr = PetscOptionsGetString(PETSC_NULLPTR, PETSC_NULLPTR, "-Ain", Ain, PETSC_MAX_PATH_LEN, &flg_A);
    A = Get_A(Ain);
}

