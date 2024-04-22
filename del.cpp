#include <petsc.h>

int main(int argc, char **argv)
{
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

    // 在这里调用其他 PETSc 函数

    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}