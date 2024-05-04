import MicroAnalysisV6 as micro
import numpy as np
import RelatedFunctions as rf
import time

if __name__ == '__main__':
    func = rf.get_TPMS_func('Strut G')
    const = rf.get_TPMS_const('Strut G',0.3)

    for mesh_size in np.arange(128,138,10):
        x,_ = rf.TPMS2Mesh(mesh_size,1,func,const)
        # stime = time.time()
        # CH = micro.homogenization3d(mesh_size,[rf.get_C0()],x,Device='cpu')
        # print('Mesh:%d,Time:%.2f,Device:cpu,C11:%.8f'%(mesh_size,time.time()-stime,CH[0,0]))
        stime = time.time()
        CH = micro.homogenization3d(mesh_size,[rf.get_C0()],x,Device='gpu',num_cores=16)
        print('Mesh:%d,Time:%.2f,Device:gpu,C11:%.8f'%(mesh_size,time.time()-stime,CH[0,0]))