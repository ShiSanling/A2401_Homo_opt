
"""
MicroAnalysis Version6
Function = to analysis lattice model performence use hm method.
修复了计算结果弹性张量C11,C22,C33相反，C44,C55,C66顺序出错，前者因为x矩阵排列错误（在read_stl_part改正），后者因为应力顺序排列错误（在B矩阵计算中改正）
Date = 2022.11.6
Author = <ZKJi>
添加CUDA求解线性方程组
Date = 2024.04.27
"""

from importlib.metadata import distribution
import numpy as np
from scipy import sparse
import time
import vtkmodules.all as vtk
import multiprocessing as mp
import os
import sys
from joblib import Parallel, delayed
import cupy as cp
from cupyx.scipy.sparse.linalg import cg

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
    print(time.time()-stime)
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

# def compute_K(edofMat, Ke, x, voxel, existEle, Emin, E0, nele):
#     iK = np.tile(edofMat, 24).flatten()
#     jK = np.repeat(edofMat, 24).flatten()
#     sK = np.zeros(len(iK))
#
#     x_current = x.copy()
#     x_current[np.where(voxel != 0)] = 0
#     x_norm = (Emin + x_current[existEle]**3 * (E0 - Emin)).flatten()
#     sK += np.kron(x_norm, Ke[0].flatten())
#
#     K = sparse.csr_matrix((sK, (iK, jK)), shape=(3 * nele, 3 * nele), dtype=np.float32)
#     K = (K + K.T) / 2
#
#     return K

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


def compute_CH(i, j, Ke, U0, U, edofMat_, voxel, x, mesh_size):
    sumCH = 0
    for k in range(len(Ke)):
        current_Ele = np.where((voxel == k) & (x > 0))
        current_ELe_line = current_Ele[0] * mesh_size ** 2 + current_Ele[2] * mesh_size + mesh_size - 1 - current_Ele[1]
        current_x_line = x[current_Ele]
        sumCHi = np.multiply((U0[current_ELe_line, :, i] - U[edofMat_[current_ELe_line], i]).dot(Ke[k]), (U0[current_ELe_line, :, j] - U[edofMat_[current_ELe_line], j]))
        sumCHi = np.sum(sumCHi, axis=1)
        sumCH += np.sum(np.multiply(current_x_line, sumCHi))
    return 1 / (mesh_size ** 3) * sumCH


def calc_KeFe_thermal(k0, length_x, length_y, length_z):
    #C0 is the elasticity tensor of base materials, which is isotropic
    # C0 = E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])

    coordinates = [[-length_x/2,-length_y/2,-length_z/2],[length_x/2,-length_y/2,-length_z/2],[length_x/2,length_y/2,-length_z/2],[-length_x/2,length_y/2,-length_z/2],[-length_x/2,-length_y/2,length_z/2],[length_x/2,-length_y/2,length_z/2],[length_x/2,length_y/2,length_z/2],[-length_x/2,length_y/2,length_z/2]]
    GaussPoint = [-1/np.sqrt(3),1/np.sqrt(3)]
    Ke = np.zeros((8,8))
    Fe = np.zeros((8,3))
    for xi1 in GaussPoint:
        for xi2 in GaussPoint:
            for xi3 in GaussPoint:
                dShape = (1/8)*np.array([[-(1-xi2)*(1-xi3),(1-xi2)*(1-xi3),(1+xi2)*(1-xi3),-(1+xi2)*(1-xi3),-(1-xi2)*(1+xi3),(1-xi2)*(1+xi3),(1+xi2)*(1+xi3),-(1+xi2)*(1+xi3)],[-(1-xi1)*(1-xi3),-(1+xi1)*(1-xi3),(1+xi1)*(1-xi3),(1-xi1)*(1-xi3),-(1-xi1)*(1+xi3),-(1+xi1)*(1+xi3),(1+xi1)*(1+xi3),(1-xi1)*(1+xi3)],[-(1-xi1)*(1-xi2),-(1+xi1)*(1-xi2),-(1+xi1)*(1+xi2),-(1-xi1)*(1+xi2),(1-xi1)*(1-xi2),(1+xi1)*(1-xi2),(1+xi1)*(1+xi2),(1-xi1)*(1+xi2)]])
                JacobianMatrix = dShape.dot(coordinates)
                B = np.linalg.inv(JacobianMatrix).dot(dShape)
                Ke = Ke + B.T.dot(k0.dot(B*np.linalg.det(JacobianMatrix)))
                Fe = Fe + B.T.dot(k0.dot(np.diag([1,1,1])*np.linalg.det(JacobianMatrix)))
    return [Ke,Fe,B]

def read_stl_part(mesh_size,filename, list_range, small_mesh_size=4,set_bound = 0):
    # x is the volume fraction on each element.

    stime = time.time()
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly_data = reader.GetOutput()
    cell_dims = [mesh_size, mesh_size, mesh_size]#x, y, z
    #x_min:0 x_max:1, y_min:2,y_max:3,z_min:4,z_max:5
    if set_bound == 0:
        bounds = poly_data.GetBounds()
    else :
        bounds = [0.1,9.9,0.1,9.9,0.1,9.9]
    mesh_pitch = [(bounds[1] - bounds[0])/cell_dims[0],
                (bounds[3] - bounds[2])/cell_dims[1],
                (bounds[5] - bounds[4])/cell_dims[2]]
    mins = [bounds[0], bounds[2], bounds[4]]
    points = vtk.vtkPoints()
    for iz in range(cell_dims[0]+1):
        for iy in range(cell_dims[1]+1):
            for ix in range(cell_dims[2]+1):
                xx = ix * mesh_pitch[0] + mins[0]
                xy = iy * mesh_pitch[1] + mins[1]
                xz = iz * mesh_pitch[2] + mins[2]
                points.InsertNextPoint(xx,xy,xz)
    structured_base_mesh = vtk.vtkStructuredGrid()
    structured_base_mesh.SetExtent(0, cell_dims[0], 0, cell_dims[1], 0, cell_dims[2])
    structured_base_mesh.SetPoints(points)

    append = vtk.vtkAppendFilter()
    append.AddInputData(structured_base_mesh)
    append.Update()
    base_mesh = append.GetOutput()

    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(base_mesh)
    cell_centers.Update()

    center_points = cell_centers.GetOutput().GetPoints()
    sdf = vtk.vtkImplicitPolyDataDistance()
    sdf.SetInput(poly_data)
    sdf.SetTolerance(10**-5)

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = np.array(mesh_pitch)/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    mesh_dot = mesh_dot * small_mesh_pitch

    results = []
    for idx in list_range:
        current_center = center_points.GetPoint(idx) #(point_x, point_y, point_z )

        distance = sdf.FunctionValue(current_center)
        # distance_sdf.InsertNextValue(distance*(mesh_size/(bounds[1])))
        if distance > np.linalg.norm(mesh_pitch)/2:
            value = 0
        elif distance < -np.linalg.norm(mesh_pitch)/2:
            value = 1
        else:
            mesh_point = current_center + mesh_dot
            value = 0
            for index in range(small_mesh_size**3):
                if sdf.FunctionValue(mesh_point[index])<0:
                    value = value + 1
            value =value/(small_mesh_size**3)
        results.append(value)
    return results

def read_stl_implicit(mesh_size,filename, list_range, small_mesh_size=4,set_bound = 0):
    # x is the volume fraction on each element.

    stime = time.time()
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly_data = reader.GetOutput()
    cell_dims = [mesh_size, mesh_size, mesh_size]#x, y, z
    #x_min:0 x_max:1, y_min:2,y_max:3,z_min:4,z_max:5
    if set_bound ==0:
        bounds = poly_data.GetBounds()
    else :
        bounds = [0.1,9.9,0.1,9.9,0.1,9.9]
    mesh_pitch = [(bounds[1] - bounds[0])/cell_dims[0],
                (bounds[3] - bounds[2])/cell_dims[1],
                (bounds[5] - bounds[4])/cell_dims[2]]
    mins = [bounds[0], bounds[2], bounds[4]]
    points = vtk.vtkPoints()
    for iz in range(cell_dims[0]+1):
        for iy in range(cell_dims[1]+1):
            for ix in range(cell_dims[2]+1):
                xx = ix * mesh_pitch[0] + mins[0]
                xy = iy * mesh_pitch[1] + mins[1]
                xz = iz * mesh_pitch[2] + mins[2]
                points.InsertNextPoint(xx,xy,xz)
    structured_base_mesh = vtk.vtkStructuredGrid()
    structured_base_mesh.SetExtent(0, cell_dims[0], 0, cell_dims[1], 0, cell_dims[2])
    structured_base_mesh.SetPoints(points)

    append = vtk.vtkAppendFilter()
    append.AddInputData(structured_base_mesh)
    append.Update()
    base_mesh = append.GetOutput()

    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(base_mesh)
    cell_centers.Update()

    center_points = cell_centers.GetOutput().GetPoints()
    sdf = vtk.vtkImplicitPolyDataDistance()
    sdf.SetInput(poly_data)

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = np.array(mesh_pitch)/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    mesh_dot = mesh_dot * small_mesh_pitch

    results = []
    for idx in list_range:
        current_center = center_points.GetPoint(idx)
        distance = sdf.FunctionValue(current_center)
        # distance_sdf.InsertNextValue(distance*(mesh_size/(bounds[1])))
        if distance > np.linalg.norm(mesh_pitch)/2:
            value = 0
        elif distance < -np.linalg.norm(mesh_pitch)/2:
            value = 1
        else:
            mesh_point = current_center + mesh_dot
            value = 0
            for index in range(small_mesh_size**3):
                if sdf.FunctionValue(mesh_point[index])<0:
                    value = value + 1
            value =value/(small_mesh_size**3)
        results.append(value)
    return results

"""
mesh_size is the number of elements divided on a single axis;
small_mesh_size is the number of elements divided on a single axis of the resampled elements;
filename is the file path of the lattice model.
"""
def read_stl(mesh_size, filename,set_bound=0, small_mesh_size=4, save_vtkfile = False, num_cores=1):
    num_cores = min(num_cores,int(mp.cpu_count()))
    pool = mp.Pool(num_cores)
    pitch =  np.ceil(mesh_size**3/num_cores).astype(int)
    x=[]
    list_range = [range(pitch*process, min(pitch*(process+1),mesh_size**3)) for process in range(num_cores)]
    results = [pool.apply_async(read_stl_part,(mesh_size,filename,list_range[process],small_mesh_size,set_bound)) for process in range(num_cores)]
    for i in range(num_cores):
        x.extend(results[i].get())
    pool.close() #关闭进程池
    pool.join()  #结束
    x = np.array(x)
    x = x.reshape((mesh_size,mesh_size,mesh_size)) #this x.shape is (z, x, y)

    if save_vtkfile:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()
        poly_data = reader.GetOutput()
        cell_dims = [mesh_size, mesh_size, mesh_size]#x, y, z
        #x_min:0 x_max:1, y_min:2,y_max:3,z_min:4,z_max:5
        if set_bound ==0:
            bounds = poly_data.GetBounds()
        else :
            bounds = [0,10,0,10,0,10]
        mesh_pitch = [(bounds[1] - bounds[0])/cell_dims[0],
                    (bounds[3] - bounds[2])/cell_dims[1],
                    (bounds[5] - bounds[4])/cell_dims[2]]
        mins = [bounds[0], bounds[2], bounds[4]]
        #点排列顺序 zyx-index,所以在此函数中注意
        points = vtk.vtkPoints()
        for iz in range(cell_dims[2]+1):
            for iy in range(cell_dims[1]+1):
                for ix in range(cell_dims[0]+1):
                    xx = ix * mesh_pitch[0] + mins[0]
                    xy = iy * mesh_pitch[1] + mins[1]
                    xz = iz * mesh_pitch[2] + mins[2]
                    points.InsertNextPoint(xx,xy,xz)
        structured_base_mesh = vtk.vtkStructuredGrid()
        structured_base_mesh.SetExtent(0, cell_dims[0], 0, cell_dims[1], 0, cell_dims[2])
        structured_base_mesh.SetPoints(points)

        append = vtk.vtkAppendFilter()
        append.AddInputData(structured_base_mesh)
        append.Update()
        base_mesh = append.GetOutput()

        cell_list = vtk.vtkIdList()
        x_volfrac = vtk.vtkDoubleArray()
        x_volfrac.SetName("volfrac")
        for idx in range(mesh_size**3):
            if x[idx//(mesh_size**2),idx//mesh_size%mesh_size,idx%mesh_size] !=0:
                cell_list.InsertNextId(idx)
            x_volfrac.InsertNextValue(x[idx//(mesh_size**2),idx//mesh_size%mesh_size,idx%mesh_size])


        base_mesh.GetCellData().SetScalars(x_volfrac)
        extract_cells = vtk.vtkExtractCells()
        extract_cells.SetInputData(base_mesh)
        extract_cells.SetCellList(cell_list)
        extract_cells.Update()

        if not os.path.exists("out"):
            os.mkdir("out")
        writer = vtk.vtkXMLDataSetWriter()
        vtk_filename = "./out/%s_volfrac.vtu"%filename.split("/")[-1].split(".")[0]
        writer.SetFileName(vtk_filename)
        writer.SetInputData(extract_cells.GetOutput())
        writer.Update()
        # print("volfrac matrix is successfully stored at %s"%filename)
    else:
        vtk_filename =""

    return x,vtk_filename


"""
mesh_size is the number of elements divided on a single axis;
small_mesh_size is the number of elements divided on a single axis of the resampled elements;
E is base materials modulus of elasticity;
nu is base materials possion's radio;
x is the volume fraction on each element.
"""
def homogenization3d(mesh_size, C0, x, voxel = None, Device = 'cpu', num_cores = 1):
    if voxel is None:
        voxel = np.zeros_like(mesh_size)
    E0=1
    Emin = 1e-3
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
    # nodeids = np.mat(np.reshape(range(0,nely*nelx),(nely,nelx))).T
    # nodeids = np.mat(np.reshape(range(0,nely*nelx),(nelx,nely))).T
    # nodeids = np.hstack((nodeids,nodeids[:,0]))
    # nodeids = np.vstack((nodeids,nodeids[0,:]))
    # nodeidz = np.hstack((range(0,(nelz-1)*nely*nelx+1,nely*nelx),[0]))
    # for i in range(nelz+1):
    #     nodeid[i] = nodeids +nodeidz[i]
    # for i in range(nele):
    #     grdz = i//(nelx*nely)
    #     grdzs = np.repeat([grdz,grdz+1],4)
    #     grdx = (i-grdz*(nelx*nely))//nely
    #     grdxs = np.hstack(([grdx,grdx+1,grdx+1,grdx],[grdx,grdx+1,grdx+1,grdx]))
    #     grdy = nely-(i-grdz*nelx*nely-grdx*nely)-1
    #     grdys = np.hstack(([grdy,grdy,grdy+1,grdy+1],[grdy,grdy,grdy+1,grdy+1]))
    #     edofMat2[i,:] = 3*np.repeat(nodeid[grdzs,nely-grdys,grdxs],3) + np.kron([1]*8,[0,1,2])
    # print(np.sum(edofMat2-edofMat))

    K = compute_K(edofMat, Ke, x, voxel, existEle, Emin, E0, nele)
    F = compute_F(edofMat, Fe, x, voxel, existEle, nele)
    U = np.zeros((ndof,6))

    #---solve U matrix------
    stime = time.time()
    if Device == 'cpu':
        if sys.platform == 'win32':
            import linalg_solve_moudle as ls
        if sys.platform == 'linux':
            import linalg_solve_moudle as ls
        U[np.setdiff1d(existDof,[0,1,2]),:] = ls.linalg_solve(K[np.setdiff1d(existDof,[0,1,2]),:][:,np.setdiff1d(existDof,[0,1,2])],F[np.setdiff1d(existDof,[0,1,2]),:])
    elif Device == 'gpu':
        U[np.setdiff1d(existDof, [0, 1, 2]), :] = linalg_solve_gpu(K[np.setdiff1d(existDof, [0, 1, 2]), :][:, np.setdiff1d(existDof, [0, 1, 2])], F[np.setdiff1d(existDof, [0, 1, 2]), :].todense())
    # print(time.time()-stime)

    Ue = np.zeros((len(Ke),24,6))
    indexe = np.hstack(([3],[6],[7],[12],[9],[10],[11],list(range(13,24))))
    for i in range(len(Ke)):
        Ue[i,indexe] = np.linalg.solve(Ke[i][indexe,:][:,indexe],Fe[i][indexe])
    U0 = np.zeros((nele,24,6))
    for j in range(len(Ke)):
        current_Ele  = np.where((voxel == j) & (x > 0))
        for i in range(6):
            U0[current_Ele[0]*mesh_size**2+current_Ele[2]*mesh_size+nely-1-current_Ele[1],:,i]=Ue[j][:,i]

    CH = np.zeros((6, 6))
    results = Parallel(n_jobs=num_cores)(
        delayed(compute_CH)(i, j, Ke, U0, U, edofMat_, voxel, x, mesh_size)
        for i in range(6)
        for j in range(6)
    )
    for i in range(6):
        for j in range(6):
            CH[i, j] = results[i * 6 + j]

    # Stress = np.zeros((6,6))
    # for idz in range(mesh_size):
    #     for idx in range(mesh_size):
    #         for idy in range(mesh_size)[::-1]:
    #             index = idz*mesh_size**2 + idx *mesh_size + mesh_size - idy -1
    #             Ue = U[edofMat_[index]]
    #             Stress = Stress + x[idz,idy,idx]*C0[voxel[idz,idy,idx].astype(int)].dot(B[voxel[idz,idy,idx].astype(int)].dot(Ue))

    # Stress = Stress/mesh_size**3
    # Stress_VM1 = np.sqrt(Stress[0,0]**2+Stress[1,0]**2+Stress[2,0]**2-Stress[0,0]*Stress[1,0]-Stress[0,0]*Stress[2,0]-Stress[1,0]*Stress[2,0]+3*(Stress[3,0]**2+Stress[4,0]**2+Stress[5,0]**2))
    # Stress_VM2 = np.sqrt(Stress[0,3]**2+Stress[1,3]**2+Stress[2,3]**2-Stress[0,3]*Stress[1,3]-Stress[0,3]*Stress[2,3]-Stress[1,3]*Stress[2,3]+3*(Stress[3,3]**2+Stress[4,3]**2+Stress[5,3]**2))
    # return CH,{'compression':Stress_VM1,'shear':Stress_VM2}
    return CH


def homogenization3d_thermal(mesh_size,k0,x):
    E0=1
    Emin = 1e-9
    [Ke,Fe,B]=calc_KeFe_thermal(k0,1,1,1)

    #calculate the arrangement of each element and the nodes.
    nelx = nely = nelz = mesh_size
    nele = nelx*nely*nelz
    ndof = 3*nele
    # nodeid = np.empty((nelz+1,nely+1,nelx+1))
    edofMat = np.empty((nele,8))
    # edofMat2 = np.empty((nele,24))
    edofids = np.mat(range(0,nely*nelx)).T
    # stime = time.time()
    edofMats = np.array(np.tile(edofids,8))+np.repeat([0,nelx*nely],4)+[1,nely+1,nely,0,1,nely+1,nely,0]
    fixMat = np.zeros((nelx*nely,8))
    for i in range(nely-1,nely*nelx,nely):
        fixMat[i,[0,1,4,5]] = fixMat[i,[0,1,4,5]]-nely
    for i in range(nelx*nely-nely,nelx*nely,1):
        fixMat[i,[1,2,5,6]] = fixMat[i,[1,2,5,6]]-nelx*nely
    edofMats = edofMats + fixMat
    edofMat = np.tile(edofMats,(nelz,1))+np.repeat(np.tile(np.array([range(0,nelz)]).T,(1,8))*nelx*nely,nelx*nely,0)
    fixMat = np.zeros((nele,8))
    for i in range(nele-nelx*nely,nele,1):
        fixMat[i,4:8] = fixMat[i,4:8]-nele
    edofMat = edofMat + fixMat
    edofMat = edofMat.astype(np.int32)
    existEle = np.where(x>0)
    edofMat_ = edofMat
    edofMat = edofMat[existEle[0]*mesh_size**2+existEle[2]*mesh_size+nely-1-existEle[1]]
    existDof = np.unique(edofMat)
    x_line = x[existEle]

    iK = np.reshape(np.kron(edofMat,np.ones((8,1))),(-1))
    jK = np.reshape(np.kron(edofMat,np.ones((1,8))),(-1))
    sK = np.reshape(np.dot(np.reshape(Ke,(8*8,1)),Emin+np.reshape(x_line,(-1,1)).T*(E0-Emin)).T,(-1))

    K = sparse.csc_matrix((sK,(iK,jK)),shape=(3*nele,3*nele),dtype=np.float32)
    K = (K+K.T)/2
    sF = np.array(np.kron(x_line.reshape(-1,1),Fe).T.reshape((1,-1)))[0]
    iF = edofMat.reshape((-1)).tolist()*3
    jF = np.hstack(([0]*8*len(existEle[0]),[1]*8*len(existEle[0]),[2]*8*len(existEle[0])))
    F = sparse.csc_matrix((sF,(iF,jF)),shape=((3*nele,3)),dtype=np.float32)
    U = np.zeros((ndof,3))
    #---solve U matrix------
    stime = time.time()
    if sys.platform == 'win32':
        import linalg_solve_moudle as ls
    if sys.platform == 'linux':
        import linalg_solve_moudle as ls
    U[np.setdiff1d(existDof,[0,1,2]),:] = ls.linalg_solve(K[np.setdiff1d(existDof,[0,1,2]),:][:,np.setdiff1d(existDof,[0,1,2])],F[np.setdiff1d(existDof,[0,1,2]),:])
    # print(time.time()-stime)

    Ue = np.zeros((8,3))
    indexe = np.setdiff1d(np.arange(8),[0])
    Ue[indexe] = np.linalg.solve(Ke[indexe,:][:,indexe],Fe[indexe])
    U0 = np.zeros((nele,8,3))
    for i in range(3):
        U0[existEle[0]*mesh_size**2+existEle[2]*mesh_size+nely-1-existEle[1],:,i]=Ue[:,i]
    KH=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            sum = np.multiply((U0[:,:,i]-U[edofMat_,i]).dot(Ke),(U0[:,:,j]-U[edofMat_,j]))
            sum = np.sum(sum,axis=1)
            KH[i,j] = 1/nele*np.sum(np.multiply(x_line,sum[existEle[0]*mesh_size**2+existEle[2]*mesh_size+nely-1-existEle[1]]))

    return KH

import RelatedFunctions as rf
if __name__=="__main__":
    # filename = "./Unit_Cell_STLs/Octet-Truss.stl"
    # mesh_size= 60
    # stime = time.time()
    # x,_ = read_stl(mesh_size,filename,4, True, num_cores = 16)
    # # x=np.ones((4,4,4))
    # # x=np.ones((mesh_size,mesh_size,mesh_size))

    # print(time.time()-stime)
    # E = 1850e6
    # nu =0.3
    # C0 = E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])
    # CH,Stress = homogenization3d(mesh_size,C0,x)
    # # RelatedFunctions.elastic_modulus_visualization2d(CH)
    # # np.save("CH.npy",CH)
    # S = np.linalg.inv(CH)
    # EH = 1/S[0,0]
    # GH = 1/S[3,3]
    # nuH = -S[0,1]/S[0,0]
    # AH = 2*CH[3,3]/(CH[0,0]-CH[0,1])
    # print("mesh_size %f time_costing:"%mesh_size,time.time()-stime,EH,GH,nuH,AH)


    # multi-material
    C0 = []
    voxel = []

    filename = "Cubic-BCC_1/0-30.stl"
    mesh_size= 10
    # x,_ = read_stl(mesh_size,filename,4, True, num_cores = 16)
    # x[np.where(x>0)] = 1
    x = np.ones((mesh_size, mesh_size, mesh_size))*0.3
    # x[mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4] = 0.3/3
    stime = time.time()
    Es = [1, 1]
    nus = [0.3, 0.3]
    for i in range(len(Es)):
        E = Es[i]
        nu = nus[i]
        C0.append( E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]]))
    # x=np.ones((4,4,4))
    # x=np.ones((mesh_size,mesh_size,mesh_size))
    k0 = np.diag([60.5,60.5,60.5])
    voxel = np.ones((mesh_size,mesh_size,mesh_size))
    voxel[mesh_size//2:mesh_size,:,:] = 1

    CH,Stress = homogenization3d(mesh_size, C0, x, voxel)
    # KH = homogenization3d_thermal(mesh_size,k0,x)
    # RelatedFunctions.elastic_modulus_visualization2d(CH)
    # np.save("CH.npy",CH)
    S = np.linalg.inv(CH)
    EH = 1/S[0,0]
    GH = 1/S[3,3]
    nuH = -S[0,1]/S[0,0]
    AH = 2*CH[3,3]/(CH[0,0]-CH[0,1])

    print("mesh_size %f time_costing:"%mesh_size,time.time()-stime,EH,GH,nuH,AH)
    # x=np.ones((4,4,4))
    # x=np.ones((mesh_size,mesh_size,mesh_size))
    k0 = np.diag([60.5,60.5,60.5])
    voxel = np.zeros((mesh_size,mesh_size,mesh_size))

    CH = homogenization3d(mesh_size, [C0], x, voxel)
    # KH = homogenization3d_thermal(mesh_size,k0,x)
    rf.elastic_modulus_visualization(CH,'out/E_bone_CH')
    rf.shear_modulus_visualization(CH,'out/G_bone_CH')
    np.savetxt("CH",CH)
    S = np.linalg.inv(CH)
    EH = 1/S[0,0]
    GH = 1/S[3,3]
    nuH = -S[0,1]/S[0,0]
    AH = 2*CH[3,3]/(CH[0,0]-CH[0,1])

    print("mesh_size %f time_costing:"%mesh_size,time.time()-stime,EH,GH,nuH,AH)