
from importlib.metadata import distribution
import numpy as np
from scipy import sparse
import time 
import vtkmodules.all as vtk
import multiprocessing as mp
import os
import RelatedFunctions as rf
import sys
# import petsc4py
# from petsc4py import PETSc
# petsc4py.init(sys.argv)
import cupy as cp
# from numba import cuda
from joblib import Parallel, delayed


"""
E is base materials modulus of elasticity;
nu is base materials possion's radio;
length_x is the length in the x-dirction;
length_y is the length in the y-dirction;
length_z is the length in the z-dirction.
"""
def calc_KeFe(C0_s, length_x, length_y, length_z):
    Kes = []
    Fes = []
    Bs = []
    for C0 in C0_s:
        #C0 is the elasticity tensor of base materials, which is isotropic
        # C0 = E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])
        coordinates = [[-length_x/2,-length_y/2,-length_z/2],[length_x/2,-length_y/2,-length_z/2],[length_x/2,length_y/2,-length_z/2],[-length_x/2,length_y/2,-length_z/2],[-length_x/2,-length_y/2,length_z/2],[length_x/2,-length_y/2,length_z/2],[length_x/2,length_y/2,length_z/2],[-length_x/2,length_y/2,length_z/2]]
        GaussPoint = [-1/np.sqrt(3),1/np.sqrt(3)]
        Ke = np.zeros((24,24))
        Fe = np.zeros((24,6))
        B_all = np.zeros((6,24))
        for xi1 in GaussPoint:
            for xi2 in GaussPoint:
                for xi3 in GaussPoint:
                    dShape = (1/8)*np.array([[-(1-xi2)*(1-xi3),(1-xi2)*(1-xi3),(1+xi2)*(1-xi3),-(1+xi2)*(1-xi3),-(1-xi2)*(1+xi3),(1-xi2)*(1+xi3),(1+xi2)*(1+xi3),-(1+xi2)*(1+xi3)],[-(1-xi1)*(1-xi3),-(1+xi1)*(1-xi3),(1+xi1)*(1-xi3),(1-xi1)*(1-xi3),-(1-xi1)*(1+xi3),-(1+xi1)*(1+xi3),(1+xi1)*(1+xi3),(1-xi1)*(1+xi3)],[-(1-xi1)*(1-xi2),-(1+xi1)*(1-xi2),-(1+xi1)*(1+xi2),-(1-xi1)*(1+xi2),(1-xi1)*(1-xi2),(1+xi1)*(1-xi2),(1+xi1)*(1+xi2),(1-xi1)*(1+xi2)]])
                    JacobianMatrix = dShape.dot(coordinates)
                    auxiliar = np.linalg.inv(JacobianMatrix).dot(dShape)
                    B=np.zeros((6,24))
                    for index in range(8):
                        """ 
                        \epsilon_{x x} \\
                        \epsilon_{y y} \\
                        \epsilon_{z z} \\
                        \gamma_{y z} \\
                        \gamma_{z x} \\
                        \gamma_{x y}
                        """
                        B[0,3*index+0]=B[4,3*index+2]=B[5,3*index+1]=auxiliar[0,index]
                        B[1,3*index+1]=B[3,3*index+2]=B[5,3*index+0]=auxiliar[1,index]
                        B[2,3*index+2]=B[3,3*index+1]=B[4,3*index+0]=auxiliar[2,index]
                    Ke = Ke + B.T.dot(C0.dot(B*np.linalg.det(JacobianMatrix)))
                    Fe = Fe + B.T.dot(C0.dot(np.diag([1,1,1,1,1,1])*np.linalg.det(JacobianMatrix)))
                    B_all = B_all + B
        Kes.append(Ke)
        Fes.append(Fe)
        Bs.append(B_all)

    return [Kes,Fes,Bs]


# def linalg_solve_petsc(K, F):
#     comm = PETSc.COMM_WORLD
#
#     K_petsc = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr, K.indices, K.data))
#     F_petsc = PETSc.Mat().createAIJ(size=F.shape, csr=(F.indptr, F.indices, F.data))
#     # F_petsc = PETSc.Mat().createDense(F.todense())
#     x_petsc = PETSc.Mat().createDense(F.todense().shape)
#
#     # x_petsc = PETSc.Vec().create(comm=comm)
#     # x_petsc.setSizes(K_petsc.getSize()[0])
#     # x_petsc.setFromOptions()
#
#     ksp = PETSc.KSP().create(comm)
#     ksp.setType(PETSc.KSP.Type.CG)
#     ksp.getPC().setType(PETSc.PC.Type.JACOBI)
#     ksp.setOperators(K_petsc)
#     ksp.getPC().setOperators(K_petsc)
#     ksp.setTolerances(rtol=1.e-6)
#
#     ksp.solve(F_petsc, x_petsc)
#     return x_petsc


# from cupyx.scipy.sparse.linalg import cg
# @cuda.jit
# def cg_solver_kernel(K_gpu_data, K_gpu_indices, K_gpu_indptr, F_gpu_data, x_gpu_data, n, tol):
#     i = cuda.grid(1)
#     K_gpu = cp.sparse.csr_matrix((K_gpu_data, K_gpu_indices, K_gpu_indptr), shape=(F_gpu_data.shape[0],F_gpu_data.shape[0]))
#     print(K_gpu.shape)
#     # if i < n:
#     #     x_gpu_data[:,i] = cg(K_gpu, F_gpu_data[:,i], tol=tol)[0]
#
#
# def linalg_solve_gpu(K, F):
#     K_gpu = cp.sparse.csr_matrix(K)
#     # K_gpu_data = K_gpu.data.get()
#     # K_gpu_indices = K_gpu.indices.get()
#     # K_gpu_indptr = K_gpu.indptr.get()
#
#     F_gpu = cp.asarray(F.todense())
#     n = F.shape[1]
#     x_all = cp.zeros(F.shape, dtype=cp.float32)
#
#     threads_per_block = 256
#     blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
#     # K_gpu = cp.sparse.csr_matrix((K_gpu.data, K_gpu.indices, K_gpu.indptr), shape=(F_gpu.shape[0], F_gpu.shape[0]))
#     cg_solver_kernel[blocks_per_grid, threads_per_block](K_gpu.data, K_gpu.indices, K_gpu.indptr, F_gpu, x_all, n, 1e-5)
#
#     return x_all.get()



from cupyx.scipy.sparse.linalg import cg
def linalg_solve_gpu(K, F):
    K_gpu = cp.sparse.csc_matrix(K)
    x_all = np.zeros(F.shape)
    stime = time.time()
    streams = [cp.cuda.Stream() for _ in range(6)]  # 创建 6 个流
    events = [None] * F.shape[1]

    for i in range(6):
        with streams[i]:
            F_gpu = cp.asarray(F[:,i])
            x_gpu = cg(K_gpu, F_gpu,tol=1e-5,callback=lambda *args: None)
            events[i] = cp.cuda.Event()
            events[i].record(streams[i])
            cp.cuda.stream.get_current_stream().synchronize()
            x_all[:, i] = x_gpu[0].get()

    for i in range(F.shape[1]):
        events[i].synchronize()  # Wait for each column to finish
    return x_all

def compute_K(edofMat, Ke, x, voxel, existEle, Emin, E0, nele):
    iK = np.tile(edofMat, 24).flatten()
    jK = np.repeat(edofMat, 24).flatten()
    sK = np.zeros(len(iK))

    for i in range(len(Ke)):
        x_current = x.copy()
        x_current[np.where(voxel != i)] = 0
        x_norm = (Emin + x_current[existEle]**3 * (E0 - Emin)).flatten()
        sK += np.kron(x_norm, Ke[i].flatten())

    K = sparse.csr_matrix((sK, (iK, jK)), shape=(3 * nele, 3 * nele), dtype=np.float32)
    K = (K + K.T) / 2

    return K


def compute_F(edofMat, Fe, x, voxel, existEle, nele):
    iF = np.tile(edofMat.reshape(-1), 6)
    jF = np.repeat(np.arange(6), 24 * len(existEle[0]))
    sF = np.zeros(len(iF))
    for i in range(len(Fe)):
        x_current = x.copy()
        x_current[np.where(voxel!=i)]=0
        sF = sF + np.array(np.kron((x_current[existEle]**3).reshape(-1,1),Fe[i]).T.reshape((1,-1)))[0]
    F = sparse.csr_matrix((sF,(iF,jF)),shape=((3*nele,6)),dtype=np.float32)
    return F


"""
mesh_size is the number of elements divided on a single axis;
small_mesh_size is the number of elements divided on a single axis of the resampled elements;
E is base materials modulus of elasticity;
nu is base materials possion's radio;
x is the volume fraction on each element.
"""
def homogenization3d(mesh_size,C0,x,voxel):
    num_cores = 16
    E0=1
    Emin = 1e-9
    [Ke,Fe,B]=calc_KeFe(C0,1,1,1)

    #calculate the arrangement of each element and the nodes.
    nelx = nely = nelz = mesh_size
    nele = nelx*nely*nelz
    ndof = 3*nele
    # nodeid = np.empty((nelz+1,nely+1,nelx+1))
    edofMat = np.empty((nele,24))
    # edofMat2 = np.empty((nele,24))
    edofids = np.mat(range(0,nely*nelx)).T
    # stime = time.time()
    edofMats = 3*np.repeat(np.array(np.tile(edofids,8))+np.repeat([0,nelx*nely],4)+[1,nely+1,nely,0,1,nely+1,nely,0],3,1)+np.kron([1]*8,[0,1,2])
    fixMat = np.zeros((nelx*nely,24))
    for i in range(nely-1,nely*nelx,nely):
        fixMat[i,[0,1,2,3,4,5,12,13,14,15,16,17]] = fixMat[i,[0,1,2,3,4,5,12,13,14,15,16,17]]-nely*3
    for i in range(nelx*nely-nely,nelx*nely,1):
        fixMat[i,[3,4,5,6,7,8,15,16,17,18,19,20]] = fixMat[i,[3,4,5,6,7,8,15,16,17,18,19,20]]-nelx*nely*3
    edofMats = edofMats + fixMat
    edofMat = np.tile(edofMats,(nelz,1))+np.repeat(np.tile(np.array([range(0,nelz)]).T,(1,24))*nelx*nely*3,nelx*nely,0)
    fixMat = np.zeros((nele,24))
    for i in range(nele-nelx*nely,nele,1):
        fixMat[i,12:24] = fixMat[i,12:24]-nele*3
    edofMat = edofMat + fixMat 
    edofMat = edofMat.astype(np.int32)
    existEle = np.where(x>0)
    edofMat_ = edofMat
    edofMat = edofMat[existEle[0]*mesh_size**2+existEle[2]*mesh_size+nely-1-existEle[1]]
    existDof = np.unique(edofMat)
    x_line = x[existEle]

    stime = time.time()
    K = compute_K(edofMat, Ke, x, voxel, existEle, Emin, E0, nele)
    print(time.time() - stime)
    stime = time.time()
    F = compute_F(edofMat, Fe, x, voxel, existEle, nele)
    print(time.time() - stime)

    U = np.zeros((ndof,6))
    K_active = K[np.setdiff1d(existDof,[0,1,2]),:][:,np.setdiff1d(existDof,[0,1,2])]
    F_active = F[np.setdiff1d(existDof,[0,1,2]),:]
    
    # with open('A.txt','w') as f:
    #     f.write('%d %d %d\n'%(K_active.shape[0],K_active.shape[1],np.sum(K_active.todense()!=0)))
    #     for index1 in range(K_active.shape[0]):
    #         for index2 in range(K_active.shape[1]):
    #             if K_active[index1,index2]!=0:
    #                 f.write('%d %d %.16f\n'%(index1,index2,K_active[index1,index2]))
    
    # with open('b.txt','w') as f:
    #     for j in range(6):
    #         for index in range(F_active.shape[0]):
    #             f.write('%d %d %.16f\n'%(index,j,F_active[index,j]))
    # np.savetxt('F.txt',F_active.todense())                    
    
    # ! ---solve U matrix------
    stime = time.time()

    # import linalg_solve_moudle as ls
    # U_result = ls.linalg_solve(K_active,F_active)
    U_result = linalg_solve_gpu(K_active, F_active.todense())

    print("Solver time costing: ", time.time()-stime)
    
    U[np.setdiff1d(existDof,[0,1,2]),:] = U_result
    # np.savetxt('U.txt', U)
    Ue = np.zeros((len(Ke),24,6))
    indexe = np.hstack(([3],[6],[7],[12],[9],[10],[11],list(range(13,24))))
    for i in range(len(Ke)):
        Ue[i,indexe] = np.linalg.solve(Ke[i][indexe,:][:,indexe],Fe[i][indexe]) 
    U0 = np.zeros((nele,24,6))
    for j in range(len(Ke)):
        current_Ele  = np.where((voxel == j) & (x > 0))
        for i in range(6):
            U0[current_Ele[0]*mesh_size**2+current_Ele[2]*mesh_size+nely-1-current_Ele[1],:,i]=Ue[j][:,i]
    stime = time.time()
    CH=np.zeros((6,6))
    results = Parallel(n_jobs=num_cores)(
        delayed(compute_CH)(i, j, Ke, U0, U, edofMat_)
        for i in range(6)
        for j in range(6)
    )
    for i in range(6):
        for j in range(6):
            CH[i, j] = results[i * 6 + j]
    print(time.time() - stime)
    return CH


def compute_CH(i, j, Ke, U0, U, edofMat_):
    sumCH = 0
    for k in range(len(Ke)):
        current_Ele = np.where((voxel == k) & (x > 0))
        current_ELe_line = current_Ele[0] * mesh_size ** 2 + current_Ele[2] * mesh_size + mesh_size - 1 - current_Ele[1]
        current_x_line = x[current_Ele]
        sumCHi = np.multiply((U0[current_ELe_line, :, i] - U[edofMat_[current_ELe_line], i]).dot(Ke[k]), (U0[current_ELe_line, :, j] - U[edofMat_[current_ELe_line], j]))
        sumCHi = np.sum(sumCHi, axis=1)
        sumCH += np.sum(np.multiply(current_x_line, sumCHi))
    return 1 / (mesh_size ** 3) * sumCH

if __name__=="__main__":
    

    # multi-material
    C0 = []
    voxel = []

    mesh_size= 160 # TODO modify this parameter
    x,_ = rf.TPMS2Mesh(mesh_size,1,rf.get_TPMS_func('Strut G'),0)
    print("voxel number:", np.sum(x))
    # x[np.where(x>0)] = 1
    # x = np.ones((mesh_size, mesh_size, mesh_size))*0.3
    # x[mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4] = 0.3/3
    stime = time.time() 
    E = 1
    nu = 0.3
    C0 = E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])
    voxel = np.zeros((mesh_size,mesh_size,mesh_size))
    CH = homogenization3d(mesh_size, [C0], x, voxel)
    # KH = homogenization3d_thermal(mesh_size,k0,x)
    # RelatedFunctions.elastic_modulus_visualization2d(CH)
    # np.save("CH.npy",CH)
    S = np.linalg.inv(CH)
    EH = 1/S[0,0]
    GH = 1/S[3,3]
    nuH = -S[0,1]/S[0,0]
    AH = 2*CH[3,3]/(CH[0,0]-CH[0,1])
    print("mesh_size %f time_costing:"%mesh_size,time.time()-stime,EH)