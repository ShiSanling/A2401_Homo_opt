static char help[] = "linear solve";
#include <petscmat.h>
#include <petscksp.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
    Mat A;
    Vec b, u, u_tmp, x;
    char Ain[PETSC_MAX_PATH_LEN], rhs[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;
    int m, n, nz, dummy;
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
    CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-Ain", Ain, PETSC_MAX_PATH_LEN, &flg_A);
    CHKERRQ(ierr);
    if (flg)
        shift = 0;
    if (flg_A)
    {
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Read matrix in ascii format ...\n");CHKERRQ(ierr);
        Afile = fopen(Ain, "r");
        fscanf(Afile, "%d %d %d\n", &m, &n, &nz);
        // ierr = PetscPrintf(PETSC_COMM_WORLD,"m: %d, n: %d, nz: %d \n", m,n,nz);CHKERRQ(ierr);
        if (m != n)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for this example\n");
        ierr = MatCreate(PETSC_COMM_WORLD, &A);
        CHKERRQ(ierr);
        ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);
        CHKERRQ(ierr);
        ierr = MatSetFromOptions(A);
        CHKERRQ(ierr);
        PetscCall(MatSetUp(A));
        ierr = MatGetOwnershipRange(A, &rstart, &rend);
        CHKERRQ(ierr);
        // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart,rend-1);
        for (i = 0; i < nz; i++)
        {
            fscanf(Afile, "%d %d %le\n", &row, &col, (double *)&val);
            if (row >= rstart && row < rend)
                ierr = MatSetValues(A, 1, &row, 1, &col, &val, INSERT_VALUES);
            CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        fflush(stdout);
        fclose(Afile);
    }
    ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-rhs", rhs, PETSC_MAX_PATH_LEN, &flg_b);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-rhs", rhs, PETSC_MAX_PATH_LEN, &flg_b);
    CHKERRQ(ierr);
    if (flg_b)
    {
        ierr = VecCreate(PETSC_COMM_WORLD, &b);
        CHKERRQ(ierr);
        ierr = VecSetSizes(b, PETSC_DECIDE, n);
        CHKERRQ(ierr);
        ierr = VecSetFromOptions(b);
        CHKERRQ(ierr);
        ierr = VecGetOwnershipRange(b, &rstart1, &rend1);
        CHKERRQ(ierr);
        // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart1,rend1-1);
        bfile = fopen(rhs, "r");
        for (i = 0; i < n; i++)
        {
            fscanf(bfile, "%d %le\n", &dummy, (double *)&val);
            if (dummy >= rstart1 && dummy < rend1)
                ierr = VecSetValues(b, 1, &i, &val, INSERT_VALUES);
            CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(b);
        CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b);
        CHKERRQ(ierr);
        fflush(stdout);
        fclose(bfile);
    }
    /*ierr = PetsfcPrint(PETSC_COMM_WORLD,"\n Write matrix in binary to 'matrix.dat' ...\n");CHKERRQ(ierr);
     ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&view);CHKERRQ(ierr);
     ierr = MatView(A,view);CHKERRQ(ierr);
     if (flg_b){
   ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write rhs in binary to 'matrix.dat' ...\n");CHKERRQ(ierr);
       ierr = VecView(b,view);CHKERRQ(ierr);
     }
     ierr = MatDestroy(&A);CHKERRQ(ierr);
     if (flg_b) {ierr = VecDestroy(&b);CHKERRQ(ierr);}
     ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);*/
    PetscCall(VecDuplicate(b, &x));
    PetscLogDouble start_time, end_time, time;
    PetscTime(&start_time);
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(ksp));
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