static char help[] = "linear solve";
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>
#include <iostream>
#include <fstream>

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
        fscanf(Afile, "%d %d %lf\n", &row, &col, (double *)&val);
        if (row >= rstart && row < rend)
            ierr = MatSetValues(A, 1, &row, 1, &col, &val, ADD_VALUES);
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);

    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    fflush(stdout);
    fclose(Afile);
    return A;
}
Vec Get_B(char *rhs, int col)
{
    Vec b;
    PetscErrorCode ierr;
    int dummy, bindex, n, nz;
    PetscInt rstart1, rend1;
    PetscScalar val;
    FILE *bfile;
    PetscMPIInt size, rank;

    std::ofstream file("output.log");
    std::streambuf *coutbuf = std::cout.rdbuf(); // 保存 std::cout 的缓冲区指针
    std::cout.rdbuf(file.rdbuf()); // 将文件的缓冲区指针绑定到 std::cout

    bfile = fopen(rhs, "r");
    fscanf(bfile, "%d %d\n", &n, &nz);
    ierr = VecCreate(PETSC_COMM_WORLD, &b);
    ierr = VecSetSizes(b, PETSC_DECIDE, n);
    ierr = VecSetFromOptions(b);
    ierr = VecGetOwnershipRange(b, &rstart1, &rend1);

    // printf("\nThis is process %d reading from %d to %d ...\n",rank,rstart1,rend1-1);

    for (int i = 0; i < nz; i++)
    {
        fscanf(bfile, "%d %d %lf\n", &dummy, &bindex, (double *)&val);
        // std::cout<<dummy<<" "<<val<<" "<<bindex<<" "<<col<<std::endl;
        if ((dummy >= rstart1 && dummy < rend1) && bindex == col)
        {
            // if(dummy==5)
            //     std::cout<<dummy<<" "<<val<<" "<<bindex<<" "<<col<<std::endl;
            ierr = VecSetValues(b, 1, &dummy, &val, ADD_VALUES);
        }

    }
    ierr = VecAssemblyBegin(b);
    ierr = VecAssemblyEnd(b);

    fflush(stdout);
    fclose(bfile);

    // 恢复 std::cout 的缓冲区指针
    std::cout.rdbuf(coutbuf);

    // 关闭文件
    file.close();

    return b;
}

void Write_Vec(Vec v,const char* filename)
{
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    VecView(v, viewer);
    PetscViewerDestroy(&viewer);
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
    // b = Get_B(rhs, 0, 2175);
    // const char b_filename[10] = "b_vet.txt";
    // Write_Vec(b, b_filename);
    // PetscScalar *array;
    // VecGetArray(b, &array);
    // PetscScalar first_element = array[0];
    // PetscPrintf(PETSC_COMM_WORLD, "The first element of Vec B is: %f\n", (double)first_element);

    // PetscInt row1 = 0;
    // PetscInt col1 = 25;
    // ierr = MatGetValues(A, 1, &row1, 1, &col1, &val);
    // PetscPrintf(PETSC_COMM_WORLD, "A矩阵的第(0, 0)元素为: %f\n", val);
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
    
    PetscLogDouble start_time, end_time, time;
    PetscTime(&start_time);
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetTolerances(ksp, 1.e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(ksp));
    PetscTime(&end_time);
    time = end_time - start_time;
    PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-7.5f seconds\n", "time:", time);
    for(int idx=0;idx<6;idx++)
    {
        b = Get_B(rhs, idx);
        PetscCall(VecDuplicate(b, &x));
        PetscInt numRowsA, numColsA;
        MatGetSize(A, &numRowsA, &numColsA);  
        PetscInt sizeB;
        VecGetSize(b, &sizeB);

        PetscCall(KSPSolve(ksp, b, x));
        Vec y;
        VecDuplicate(x, &y);
        MatMult(A, x, y);

        const char y_filename[10] = "y_vec.txt";
        Write_Vec(y, y_filename);

        const char b_filename[10] = "b_vet.txt";
        Write_Vec(b, b_filename);
        std::string filename = "output"+std::to_string(idx)+".txt";
        Write_Vec(x, filename.c_str());

        PetscScalar *array;
        VecGetArray(x, &array);
        PetscScalar first_element = array[0];
        PetscPrintf(PETSC_COMM_WORLD, "The first element of Vec x is: %f\n", (double)first_element);

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

// mpiexec -n 2 ./ksp -Ain A.txt -rhs b.txt