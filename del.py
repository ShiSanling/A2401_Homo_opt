import vtk
import pyvista as pv
import numpy as np
import time
import MicroAnalysisV5 as micro
import RelatedFunctions as rf

if __name__ == '__main__':
    # multi-material
    C0 = []

    mesh_size= 10 # TODO modify this parameter
    x,_ = rf.TPMS2Mesh(mesh_size,1,rf.get_TPMS_func('Strut G'),0)
    # x[np.where(x>0)] = 1
    # x = np.ones((mesh_size, mesh_size, mesh_size))*0.3
    # x[mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4, mesh_size//4:3*mesh_size//4] = 0.3/3
    print(np.max(x))
    stime = time.time()
    E = 1
    nu = 0.3
    C0 = E*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])
    voxel = np.zeros((mesh_size,mesh_size,mesh_size))
    CH = micro.homogenization3d(mesh_size, [C0], x, voxel)
    # KH = homogenization3d_thermal(mesh_size,k0,x)
    # RelatedFunctions.elastic_modulus_visualization2d(CH)
    # np.save("CH.npy",CH)
    S = np.linalg.inv(CH)
    EH = 1/S[0,0]
    GH = 1/S[3,3]
    nuH = -S[0,1]/S[0,0]
    AH = 2*CH[3,3]/(CH[0,0]-CH[0,1])
    print("mesh_size %f time_costing:"%mesh_size,time.time()-stime,EH)