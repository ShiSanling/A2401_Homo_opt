static char help[] = "linear solve";
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>
#include <iostream>

Mat Get_A(char *Ain)
{
    Mat A;
    PetscErrorCode ierr;
    PetscInt row, col, rstart, rend;
    int m, n, nz, dummy, bindex;
    PetscScalar val;
    FILE *Afile;
    PetscMPIInt size, rank;

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
    return A;
}
Vec Get_B(char *rhs, PetscInt col, int n)
{
    Vec b;
    PetscErrorCode ierr;
    int dummy, bindex;
    PetscInt rstart1, rend1;
    PetscScalar val;
    FILE *bfile;
    PetscMPIInt size, rank;

    ierr = VecCreate(PETSC_COMM_WORLD, &b);
    ierr = VecSetSizes(b, PETSC_DECIDE, n);
    ierr = VecSetFromOptions(b);
    ierr = VecGetOwnershipRange(b, &rstart1, &rend1);

    // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart1,rend1-1);
    bfile = fopen(rhs, "r");
    for (int i = 0; i < n; i++)
    {
        fscanf(bfile, "%d %d %le\n", &dummy, &bindex, (double *)&val);
        if ((dummy >= rstart1 && dummy < rend1) && bindex == col)
            ierr = VecSetValues(b, 1, &dummy, &val, INSERT_VALUES);
    }
    ierr = VecAssemblyBegin(b);
    ierr = VecAssemblyEnd(b);

    fflush(stdout);
    fclose(bfile);
    return b;
}

int main(int argc, char **args)
{
    Mat A;
    Vec b, x, u, u_tmp;
    char Ain[PETSC_MAX_PATH_LEN], rhs[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;
    int m, n, nz, dummy, bindex;
    PetscInt i, col, row, rstart, rend, rstart1, rend1;
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
    
    ierr = PetscOptionsGetString(PETSC_NULLPTR, PETSC_NULLPTR, "-rhs", rhs, PETSC_MAX_PATH_LEN, &flg_b);
    std::cout<<1<<std::endl;
    A = Get_A(Ain);
    b = Get_B(rhs, 0, 3000);

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
    PetscTime(&end_time);
    time = end_time - start_time;
    PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-7.5f seconds\n", "time:", time);
    for(int idx=0;idx<6;idx++)
    {
        b = Get_B(rhs, idx, 3000);
        PetscInt numRowsA, numColsA;
        MatGetSize(A, &numRowsA, &numColsA);  
        PetscInt sizeB;
        VecGetSize(b, &sizeB);

        std::cout<<numRowsA<<numColsA<<sizeB<<std::endl;
        PetscCall(KSPSolve(ksp, b, x));
        PetscViewer viewer;
        std::string filename = "output"+std::to_string(idx)+".txt";
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
        VecView(x, viewer);
        PetscViewerDestroy(&viewer);
        
        PetscTime(&end_time);
        time = end_time - start_time;
        PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-7.5f seconds\n", "time:", time);
    }
        
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

// mpiexec -n 10 ./ksp -Ain A.txt -rhs b.txt