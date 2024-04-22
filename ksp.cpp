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
    
    PetscCall(MatSetUp(A));
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
Vec get_B(char* rhs, PetscInt col)
{
    Vec b;
    PetscErrorCode ierr;
    int n, dummy, bindex;
    PetscInt rstart1,rend1;
    PetscScalar val;
    FILE *bfile;
    PetscMPIInt size, rank;

    PetscInitialize(NULL, NULL, NULL, NULL);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    

    ierr = VecCreate(PETSC_COMM_WORLD, &b);
    
    ierr = VecSetSizes(b, PETSC_DECIDE, n);
    
    ierr = MatSetFromOptions(b);
    
    PetscCall(MatSetUp(b));
    ierr = MatGetOwnershipRange(b, &rstart1, &rend1);
    
    // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart1,rend1-1);
    bfile = fopen(rhs, "r");
    for (int i = 0; i < n; i++)
    {
        fscanf(bfile, "%d %d %le\n", &dummy, &bindex, (double *)&val);
        if (dummy >= rstart1 && dummy < rend1) && bindex==
            ierr = MatSetValues(b, 1, &dummy, 1, &bindex, &val, INSERT_VALUES);
        
    }
    ierr = MatAssemblyBegin(b, MAT_FINAL_ASSEMBLY);
    
    ierr = MatAssemblyEnd(b, MAT_FINAL_ASSEMBLY);
    
    fflush(stdout);
    fclose(bfile);
    PetscFinalize();
    return b;
}

int main(int argc, char **args)
{
    Mat A, b, x;
    Vec  u, u_tmp;
    char Ain[PETSC_MAX_PATH_LEN], rhs[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;
    int m, n, nz, dummy, bindex;
    PetscInt i, col, row, rstart, rend, rstart1, rend1;
    ;
    PetscScalar val;
    FILE *Afile, *bfile;
    PetscViewer view;
    PetscBool flg_A, flg_b, flg;
    PetscMPIInt size, rank;
    int shift;
    KSP ksp;
    PC pc;
    PetscInitialize(&argc, &args, (char *)0, help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    
    ierr = PetscOptionsGetString(PETSC_NULLPTR, PETSC_NULLPTR, "-Ain", Ain, PETSC_MAX_PATH_LEN, &flg_A);
    
    A = Get_A(Ain);
    ierr = PetscOptionsGetString(PETSC_NULLPTR, PETSC_NULLPTR, "-rhs", rhs, PETSC_MAX_PATH_LEN, &flg_b);
    b = get_B(rhs,0);
    

    /*ierr = PetsfcPrint(PETSC_COMM_WORLD,"\n Write matrix in binary to 'matrix.dat' ...\n");
     ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&view);
     ierr = MatView(A,view);
     if (flg_b){
   ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write rhs in binary to 'matrix.dat' ...\n");
       ierr = VecView(b,view);
     }
     ierr = MatDestroy(&A);
     if (flg_b) {ierr = VecDestroy(&b);}
     ierr = PetscViewerDestroy(&view);*/
    PetscCall(MatDuplicate(b,MAT_DO_NOT_COPY_VALUES,&x););
    PetscLogDouble start_time, end_time, time;
    PetscTime(&start_time);
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(ksp));
    PetscTime(&end_time);
    time = end_time - start_time;
    PetscPrintf("time:", time);
    PetscCall(KSPSolve(ksp, b, x));
    PetscTime(&end_time);
    time = end_time - start_time;
    PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-7.5f seconds\n", "Average time:", time);
    // PetscCall( VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&A));
    ierr = PetscFinalize();
    return 0;
}