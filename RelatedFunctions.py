import numpy as np
import vtk
from pathlib import Path
import os
from numpy import cos, sin, array, pi
from matplotlib import pyplot as plt
from matplotlib import ticker as tik
import math
from scipy import optimize
import scipy
import random
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersHybrid import vtkImplicitModeller
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
import sys
import time
import pyvista as pv
import base64
import multiprocessing

def inverse66(M66):
    """
    Invert tensor given in Voigt notation.

    Parameters
    ----------
    M66 : numpy.array (6,6)
        Tensor in Voigt notation.

    Returns
    -------
    I66 : numpy.array (6,6)
        Inverse of M66 in Voigt notation.

    References
    ----------
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.622.4732&rep=rep1&type=pdf

    """
    W = np.identity(6)
    W[3,3] = W[4,4] = W[5,5] = 0.5
    return np.einsum('ij,jk,kl',W,np.linalg.inv(M66),W)

def C66toC3333(stiffness):
    """
    Expand stiffness tensor from contracted Voigt notation to full rank four representation.

    Parameters
    ----------
    stiffness : numpy.array (6,6)
        Stiffness tensor in Voigt notation.

    Returns
    -------
    C3333 : numpy.array (3,3,3,3)
        Rank four stiffness tensor.

    """
    index = np.array([[0,0],[1,1],[2,2,],[1,2],[0,2],[0,1]])
    C3333 = np.zeros((3,3,3,3))
    for a in range(6):
      i = index[a][0]
      j = index[a][1]
      for b in range(6):
        k = index[b][0]
        l = index[b][1]
        C3333[i,j,k,l] = stiffness[a,b]
        C3333[i,j,l,k] = stiffness[a,b]
        C3333[j,i,k,l] = stiffness[a,b]
        C3333[j,i,l,k] = stiffness[a,b]

    return C3333

def SierpinskySpherical(t,N,nodeChild,node):
    # Subdivide the triangle and normalize the new points
    # thus generated to lie on the surface of the unit sphere.
    # input triangle with vertices labeled [0,1,2] as shown
    # below will be turned into four new triangles:
    #
    #            Make new (auto normalized) points
    #                 a = (0+1)/2
    #                 b = (1+2)/2
    #                 c = (2+0)/2
    #       C=2
    #       /\
    #      /  \
    #    c/____\ b    Construct new triangles
    #    /\    /\       t1 [0,a,c]
    #   /  \  /  \      t2 [a,1,b]
    #  /____\/____\     t3 [c,b,2]
    # 0=A    a   B=1    t4 [a,b,c]

    if N > 0:
      a = indexOfChild(t[[0,1]],nodeChild,node)
      b = indexOfChild(t[[1,2]],nodeChild,node)
      c = indexOfChild(t[[2,0]],nodeChild,node)

      return np.vstack((SierpinskySpherical(np.array([t[0],a,c]),N-1,nodeChild,node),SierpinskySpherical(np.array([  a,t[1],b]),N-1,nodeChild,node),SierpinskySpherical(np.array([c,b,t[2]]),N-1,nodeChild,node),SierpinskySpherical(np.array([a,b,c]),N-1,nodeChild,node),))
    else:
      return t

def indexOfChild(parents,nodeChild,node):
    child = '{}+{}'.format(str(np.min(parents)),str(np.max(parents)))
    if child not in nodeChild:
      nodeChild[child] = len(node) # find next highest index
      node.resize((nodeChild[child]+1,3),refcheck=False)
      node[nodeChild[child]] = np.average(node[parents],axis=0)                 # average of both parents
      node[nodeChild[child]] /= np.linalg.norm(node[nodeChild[child]])          # normalize to unit sphere

    return nodeChild[child]

def E_hkl3333(S3333,dir):

    return 1./np.einsum('i,j,k,l,ijkl',dir,dir,dir,dir,S3333)

def vtk_writeData(filename,node,connectivity):
    """
    Write a VTK PolyData object of the directional elastic modulus.

    Parameters
    ----------
    filename : str
        Name of output file. Extension will be replaced by VTK default.

    """
    polydata = vtk.vtkPolyData()
    triangles = vtk.vtkCellArray()
    triangle = vtk.vtkTriangle()
    magnitude = vtk.vtkDoubleArray()
    magnitude.SetNumberOfComponents(1)
    magnitude.SetName("E(Pa)")

    points = vtk.vtkPoints()
    for p in node:
        points.InsertNextPoint(*p)
        magnitude.InsertNextValue(np.linalg.norm(p))
        polydata.GetPointData().SetScalars(magnitude)

    for t in connectivity:
        for c in range(3):
            triangle.GetPointIds().SetId(c, t[c])
        triangles.InsertNextCell(triangle)

    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename+'.'+writer.GetDefaultFileExtension())
    writer.SetInputData(polydata)
    writer.Write()

def elastic_modulus_visualization(CH, vtp_filename, N=7):
    elastic = Elastic(np.array(CH))
    node = np.array( [
        [ 1.0, 0.0, 0.0], # 0
        [-1.0, 0.0, 0.0], # 1
        [ 0.0, 1.0, 0.0], # 2
        [ 0.0,-1.0, 0.0], # 3
        [ 0.0, 0.0, 1.0], # 4
        [ 0.0, 0.0,-1.0]  # 5
        ] )

    octahedron = np.array( [
        [ 0, 2, 4 ],
        [ 2, 1, 4 ],
        [ 1, 3, 4 ],
        [ 3, 0, 4 ],
        [ 0, 5, 2 ],
        [ 2, 5, 1 ],
        [ 1, 5, 3 ],
        [ 3, 5, 0 ],
        ] )

    S3333 = C66toC3333(inverse66(CH))

    nodeChild = {}
    for i in range(len(node)):
        nodeChild['{clone}+{clone}'.format(clone=str(i))] = i

    connectivity = np.vstack([SierpinskySpherical(t,N,nodeChild,node) for t in octahedron])

    for i,n in enumerate(node):
        # if np.abs(n[0]) <1e-9:
        #     node[i] *= elastic.Young([np.arccos(n[2]),np.pi/2])
        # else:
        #     node[i] *= elastic.Young([np.arccos(n[2]),np.arctan(n[1]/n[0])])
        node[i] *= elastic.Young([np.arccos(n[2]),np.angle(complex(n[0], n[1]))])

    vtk_writeData(vtp_filename,node,connectivity)

def shear_modulus_visualization(CH, vtp_filename, N=7):
    elastic = Elastic(np.array(CH))
    node = np.array( [
        [ 1.0, 0.0, 0.0], # 0
        [-1.0, 0.0, 0.0], # 1
        [ 0.0, 1.0, 0.0], # 2
        [ 0.0,-1.0, 0.0], # 3
        [ 0.0, 0.0, 1.0], # 4
        [ 0.0, 0.0,-1.0]  # 5
        ] )

    octahedron = np.array( [
        [ 0, 2, 4 ],
        [ 2, 1, 4 ],
        [ 1, 3, 4 ],
        [ 3, 0, 4 ],
        [ 0, 5, 2 ],
        [ 2, 5, 1 ],
        [ 1, 5, 3 ],
        [ 3, 5, 0 ],
        ] )

    S3333 = C66toC3333(inverse66(CH))

    nodeChild = {}
    for i in range(len(node)):
        nodeChild['{clone}+{clone}'.format(clone=str(i))] = i

    connectivity = np.vstack([SierpinskySpherical(t,N,nodeChild,node) for t in octahedron])
    node2 = node.copy()
    for i,n in enumerate(node):
        # if np.abs(n[0]) <1e-9:
        #     # node[i] *= elastic.shear3D(np.arccos(n[2]),np.pi/2)[0]
        #     node2[i] *= elastic.shear3D(np.arccos(n[2]),np.pi/2)[1]
        # else:
            # node[i] *= elastic.shear3D(np.arccos(n[2]),np.arctan(n[1]/n[0]))[0]
        node2[i] *= elastic.shear3D(np.arccos(n[2]),np.angle(complex(n[0], n[1])))[1]

    vtk_writeData(vtp_filename,node2,connectivity)

def dirVec(theta, phi):
    return [ math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta) ]

def dirVec2(theta, phi, chi):
    return [ math.cos(theta)*math.cos(phi)*math.cos(chi) - math.sin(phi)*math.sin(chi),
             math.cos(theta)*math.sin(phi)*math.cos(chi) + math.cos(phi)*math.sin(chi),
             - math.sin(theta)*math.cos(chi) ]

class Elastic:
    """An elastic tensor, along with methods to access it"""

    def __init__(self, mat):
        self.CVoigt = mat

        # Put it in a more useful representation
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")

        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        def SVoigtCoeff(p,q): return 1. / ((1+p//3)*(1+q//3))

        self.Smat = [[[[ SVoigtCoeff(VoigtMat[i][j], VoigtMat[k][l]) * self.SVoigt[VoigtMat[i][j]][VoigtMat[k][l]]
                         for i in range(3) ] for j in range(3) ] for k in range(3) ] for l in range(3) ]
        return

    def isOrthorhombic(self):
        def iszero(x): return (abs(x) < 1)
        return (iszero(self.CVoigt[0][3]) and iszero(self.CVoigt[0][4]) and iszero(self.CVoigt[0][5])
                and iszero(self.CVoigt[1][3]) and iszero(self.CVoigt[1][4]) and iszero(self.CVoigt[1][5])
                and iszero(self.CVoigt[2][3]) and iszero(self.CVoigt[2][4]) and iszero(self.CVoigt[2][5])
                and iszero(self.CVoigt[3][4]) and iszero(self.CVoigt[3][5]) and iszero(self.CVoigt[4][5]))

    def isCubic(self):
        def iszero(x): return (abs(x) < 1.e-3)
        return (iszero(self.CVoigt[0][3]) and iszero(self.CVoigt[0][4]) and iszero(self.CVoigt[0][5])
                and iszero(self.CVoigt[1][3]) and iszero(self.CVoigt[1][4]) and iszero(self.CVoigt[1][5])
                and iszero(self.CVoigt[2][3]) and iszero(self.CVoigt[2][4]) and iszero(self.CVoigt[2][5])
                and iszero(self.CVoigt[3][4]) and iszero(self.CVoigt[3][5]) and iszero(self.CVoigt[4][5])
                and iszero(self.CVoigt[0][0] - self.CVoigt[1][1]) and iszero(self.CVoigt[0][0] - self.CVoigt[2][2])
                and iszero(self.CVoigt[0][0] - self.CVoigt[1][1]) and iszero(self.CVoigt[0][0] - self.CVoigt[2][2])
                and iszero(self.CVoigt[3][3] - self.CVoigt[4][4]) and iszero(self.CVoigt[3][3] - self.CVoigt[5][5])
                and iszero(self.CVoigt[0][1] - self.CVoigt[0][2]) and iszero(self.CVoigt[0][1] - self.CVoigt[1][2]))

    def Young(self, x):
        a = dirVec(x[0], x[1])
        r = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/r

    def Young_2(self,x,y):
        a = dirVec(x, y)
        r = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/r

    def LC(self, x):
        a = dirVec(x[0], x[1])
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1000 * r

    def LC_2(self, x, y):
        a = dirVec(x, y)
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1000 * r

    def shear(self, x):
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r = sum([ a[i]*b[j]*a[k]*b[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/(4*r)

    def Poisson(self, x):
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r1 = sum([ a[i]*a[j]*b[k]*b[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        r2 = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return -r1/r2

    def averages(self):
        A = (self.CVoigt[0][0] + self.CVoigt[1][1] + self.CVoigt[2][2]) / 3
        B = (self.CVoigt[1][2] + self.CVoigt[0][2] + self.CVoigt[0][1]) / 3
        C = (self.CVoigt[3][3] + self.CVoigt[4][4] + self.CVoigt[5][5]) / 3
        a = (self.SVoigt[0][0] + self.SVoigt[1][1] + self.SVoigt[2][2]) / 3
        b = (self.SVoigt[1][2] + self.SVoigt[0][2] + self.SVoigt[0][1]) / 3
        c = (self.SVoigt[3][3] + self.SVoigt[4][4] + self.SVoigt[5][5]) / 3

        KV = (A + 2*B) / 3
        GV = (A - B + 3*C) / 5

        KR = 1 / (3*a + 6*b)
        GR = 5 / (4*a - 4*b + 3*c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return [ [KV, 1/(1/(3*GV) + 1/(9*KV)), GV, (1 - 3*GV/(3*KV+GV))/2],
                 [KR, 1/(1/(3*GR) + 1/(9*KR)), GR, (1 - 3*GR/(3*KR+GR))/2],
                 [KH, 1/(1/(3*GH) + 1/(9*KH)), GH, (1 - 3*GH/(3*KH+GH))/2] ]

    def shear2D(self, x):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.shear([x[0], x[1], z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([x[0], x[1], z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def shear3D(self, x, y, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.shear([x, y, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([x, y, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))

    def Poisson2D(self, x):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.Poisson([x[0], x[1], z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([x[0], x[1], z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun))

    def poisson3D(self, x, y, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.Poisson([x, y, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([x, y, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun), float(r1.x), float(r2.x))

def minimize(func, dim):
    if dim == 2:
        r = ((0, np.pi), (0, np.pi))
        n = 25
    elif dim == 3:
        r = ((0, np.pi), (0, np.pi), (0, np.pi))
        n = 10

    return optimize.brute(func, r, Ns = n, full_output = True, finish = optimize.fmin)[0:2]


def maximize(func, dim):
    res = minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])

def figure_init():
    plt.close()
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(polar=True)
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    ax.set_rlabel_position(45)
    plt.thetagrids(np.arange(0, 360, 30), size=10)
    ax.spines['polar'].set_visible(False)
    ax.grid(linestyle=':', alpha=0.6, linewidth=2, color='black')
    return fig, ax

def plot_ploar_2d(dataU, *dataR, filename="unnamed.png", maxrad, legend = "", linewidth = 2):
    fig, ax = figure_init()
    plt.title(legend)
    Young_color = (55 / 255., 126 / 255., 184 / 255.)
    for i in range(len(dataR)):
        ax.plot(dataU, dataR[i], '-', color=Young_color, linewidth=linewidth)
    ax.set_rlim(0, maxrad)
    frmtr = tik.FormatStrFormatter("%d Pa")
    ax.yaxis.set_major_formatter(frmtr)
    plt.savefig("out/"+ filename, format='png', bbox_inches='tight', dpi=600)

def makePolarPlot(func, maxrad, legend='', npoints=1000, color= (55 / 255., 126 / 255., 184 / 255.), linewidth=2, filename="unnamed.png"):

    u = np.linspace(0, 2*np.pi, npoints)
    r=[]
    r.append(list(map(func, u)))
    fig, ax = figure_init()
    plt.title(legend)
    for i in range(len(r)):
        ax.plot(u, r[i], '-', color=color, linewidth=linewidth)
    ax.set_rlim(0, maxrad)
    frmtr = tik.FormatStrFormatter("%d Pa")
    ax.yaxis.set_major_formatter(frmtr)
    plt.savefig("out/"+ filename, format='png', bbox_inches='tight', dpi=600)

def makePolarPlotPosNeg(func, maxrad, legend='', p='xy', width=300, height=300, npoints=90, linewidth=2):
    i = random.randint(0, 100000)
    print('<div class="plot">')
    print('<div id="box%d" class="jxgbox" style="width: %dpx; height: %dpx; display:inline-block;"></div>' % (i, width, height))
    print('<br />%s</div>' % legend)
    print('<script type="text/javascript">')
    print('var b = JXG.JSXGraph.initBoard(\'box%d\', {boundingbox: [-%f, %f, %f, -%f], axis:true, showcopyright: 0});'
          % (i, maxrad, maxrad, maxrad, maxrad))

    u = np.linspace(0, np.pi, npoints)
    r1 = list(map(lambda x: max(0, func(x)), u))
    r2 = list(map(lambda x: max(0, -func(x)), u))
    plot_ploar_2d(u, r1, r2, filename = 'LC1.png', maxrad = maxrad, legend = legend, linewidth=linewidth)


def makePolarPlot2(func, maxrad, legend='', p='xy', width=300, height=300, npoints=61, linewidth=2):
    i = random.randint(0, 100000)
    print('<div class="plot">')
    print('<div id="box%d" class="jxgbox" style="width: %dpx; height: %dpx; display:inline-block;"></div>' % (i, width, height))
    print('<br />%s</div>' % legend)
    print('<script type="text/javascript">')
    print('var b = JXG.JSXGraph.initBoard(\'box%d\', {boundingbox: [-%f, %f, %f, -%f], axis:true, showcopyright: 0});'
          % (i, maxrad, maxrad, maxrad, maxrad))

    u = np.linspace(0, np.pi, npoints)
    r = list(map(func, u))

    if (p=="xy"):
        x1 = np.array([ ir[0] * np.cos(iu) for ir, iu in zip(r,u) ])
        y1 = np.array([ ir[0] * np.sin(iu) for ir, iu in zip(r,u) ])
        x2 = np.array([ ir[1] * np.cos(iu) for ir, iu in zip(r,u) ])
        y2 = np.array([ ir[1] * np.sin(iu) for ir, iu in zip(r,u) ])
    else:
        y1 = np.array([ ir[0] * np.cos(iu) for ir, iu in zip(r,u) ])
        x1 = np.array([ ir[0] * np.sin(iu) for ir, iu in zip(r,u) ])
        y2 = np.array([ ir[1] * np.cos(iu) for ir, iu in zip(r,u) ])
        x2 = np.array([ ir[1] * np.sin(iu) for ir, iu in zip(r,u) ])

    writePolarPlotData (x1, y1, "1")
    writePolarPlotData (x2, y2, "2")
    print("b.create('curve', [dataX1,dataY1], {strokeColor:'green', strokeWidth: %d});" % (linewidth))
    print("b.create('curve', [dataX2,dataY2], {strokeColor:'blue', strokeWidth: %d});" % (linewidth))
    print('</script>')

def makePolarPlot3(func, maxrad, legend='', p='xy', width=300, height=300, npoints=61, linewidth=2):
    i = random.randint(0, 100000)
    print('<div class="plot">')
    print('<div id="box%d" class="jxgbox" style="width: %dpx; height: %dpx; display:inline-block;"></div>' % (i, width, height))
    print('<br />%s</div>' % legend)
    print('<script type="text/javascript">')
    print('var b = JXG.JSXGraph.initBoard(\'box%d\', {boundingbox: [-%f, %f, %f, -%f], axis:true, showcopyright: 0});'
          % (i, maxrad, maxrad, maxrad, maxrad))

    u = np.linspace(0, np.pi, npoints)
    r = list(map(func, u))

    if (p=="xy"):
        x1 = np.array([ ir[0] * np.cos(iu) for ir, iu in zip(r,u) ])
        y1 = np.array([ ir[0] * np.sin(iu) for ir, iu in zip(r,u) ])
        x2 = np.array([ ir[1] * np.cos(iu) for ir, iu in zip(r,u) ])
        y2 = np.array([ ir[1] * np.sin(iu) for ir, iu in zip(r,u) ])
        x3 = np.array([ ir[2] * np.cos(iu) for ir, iu in zip(r,u) ])
        y3 = np.array([ ir[2] * np.sin(iu) for ir, iu in zip(r,u) ])
    else:
        y1 = np.array([ ir[0] * np.cos(iu) for ir, iu in zip(r,u) ])
        x1 = np.array([ ir[0] * np.sin(iu) for ir, iu in zip(r,u) ])
        y2 = np.array([ ir[1] * np.cos(iu) for ir, iu in zip(r,u) ])
        x2 = np.array([ ir[1] * np.sin(iu) for ir, iu in zip(r,u) ])
        y3 = np.array([ ir[2] * np.cos(iu) for ir, iu in zip(r,u) ])
        x3 = np.array([ ir[2] * np.sin(iu) for ir, iu in zip(r,u) ])

    writePolarPlotData (x1, y1, "1")
    writePolarPlotData (x2, y2, "2")
    writePolarPlotData (x3, y3, "3")
    print("b.create('curve', [dataX1,dataY1], {strokeColor:'red', strokeWidth: %d});" % (linewidth))
    print("b.create('curve', [dataX2,dataY2], {strokeColor:'green', strokeWidth: %d});" % (linewidth))
    print("b.create('curve', [dataX3,dataY3], {strokeColor:'blue', strokeWidth: %d});" % (linewidth))
    print('</script>')

def elastic_modulus_visualization2d(matrix):

    elas = Elastic(matrix)
    minE = minimize(elas.Young, 2)
    maxE = maximize(elas.Young, 2)
    minLC = minimize(elas.LC, 2)
    maxLC = maximize(elas.LC, 2)
    minG = minimize(elas.shear, 3)
    maxG = maximize(elas.shear, 3)
    minNu = minimize(elas.Poisson, 3)
    maxNu = maximize(elas.Poisson, 3)

    m = 1.2 * maxE[1]
    makePolarPlot(lambda x: elas.Young([np.pi / 2, x]), m, "Young's modulus in {100} plane", color = (55 / 255., 126 / 255., 184 / 255.),filename = "E100.png")
    makePolarPlot(lambda x: elas.Young([x, np.pi / 4]), m, "Young's modulus in {110} plane", color = (55 / 255., 126 / 255., 184 / 255.),filename = "E110.png")
    makePolarPlot(lambda x: elas.Young([-np.pi / 4, x]), m, "Young's modulus in {111} plane", color = (55 / 255., 126 / 255., 184 / 255.),filename = "E111.png")
    return "out/E100.png","out/E110.png","out/E111.png"
    # m = 1.2 * max(maxLC[1], abs(minLC[1]))
    # makePolarPlotPosNeg(lambda x: elas.LC([np.pi / 2, x]), m, "linear compressibility in (xy) plane", "xy")
    # makePolarPlotPosNeg(lambda x: elas.LC([x, 0]), m, "linear compressibility in (xz) plane", "xz")
    # makePolarPlotPosNeg(lambda x: elas.LC([x, np.pi / 2]), m, "linear compressibility in (yz) plane", "yz")

    # m = 1.2 * maxG[1]
    # makePolarPlot2(lambda x: elas.shear2D([np.pi / 2, x]), m, "Shear modulus in (xy) plane", "xy")
    # makePolarPlot2(lambda x: elas.shear2D([x, 0]), m, "Shear modulus in (xz) plane", "xz")
    # makePolarPlot2(lambda x: elas.shear2D([x, np.pi / 2]), m, "Shear modulus in (yz) plane", "yz")

    # m = 1.2 * max(abs(maxNu[1]), abs(minNu[1]))
    # makePolarPlot3(lambda x: elas.Poisson2D([np.pi / 2, x]), m, "Poisson's ratio in (xy) plane", "xy")
    # makePolarPlot3(lambda x: elas.Poisson2D([x, 0]), m, "Poisson's ratio in (xz) plane", "xz")
    # makePolarPlot3(lambda x: elas.Poisson2D([x, np.pi / 2]), m, "Poisson's ratio in (yz) plane", "yz")


def radar_plot(data,reference_data):

    # #用于正常显示符号
    # plt.rcParams['axes.unicode_minus'] = False

    plt.style.use('ggplot')

    # 构造数据
    feature = [r'$E^H$',r'$G^H$',r'$\nu^H$',r'$\sigma_{VM}$',r'$\tau_{VM}$']
    displaydata = np.array(data)/np.array(reference_data)
    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles=np.linspace(0, 2*np.pi,len(data), endpoint=False)

    # 拼接数据首尾，使图形中线条封闭
    displaydata=np.concatenate((displaydata,[displaydata[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    feature=np.concatenate((feature,[feature[0]]))
    # 绘图
    fig=plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    # 绘制折线图
    ax.plot(angles, displaydata, 'o-', linewidth=2)
    ax.tick_params(labelsize=23)
    for i in range(len(data)):
        if i != 2:
            data[i] = data[i]/10**9
        ax.annotate('{:.3g}'.format(data[i]),
                xy=(angles[i], min(np.max(displaydata),0.1+displaydata[i])),  # 被注释点遵循极坐标系，坐标为角度和半径
                xytext=(angles[i], min(np.max(displaydata),0.1+displaydata[i])),    # 注释文本放在绘图区的0.05百分比处
                fontsize = 21,
                )
    # 填充颜色
    ax.fill(angles, displaydata, alpha=0.25)

    # 设置图标上的角度划分刻度，为每个数据点处添加标签
    ax.set_thetagrids(angles * 180/np.pi, feature)

    # 设置雷达图的范围
    ax.set_ylim(0,1.1*np.max(displaydata))
    # 添加标题
    # plt.title('活动前后员工状态表现')
    # 添加网格线
    ax.grid(True)
    plt.tick_params(labelsize=21)
    plt.savefig("out/radar.png", format='png', bbox_inches='tight', dpi=200)
    # plt.show()
    return "out/radar.png"

#a:\alpha b:\beta c:(c); Euler angles(\alpha,\beta,\gamma)
def Rotate_CH(CH, Euler_angles):
    #按照xzx旋转方式，不同绕轴具有不同的R矩阵
    alpha,beta,gamma = Euler_angles.values()

    R = np.mat([[1,0,0],[0,cos(gamma),-sin(gamma)],[0,sin(gamma),cos(gamma)]])*np.mat([[cos(beta),-sin(beta),0],[sin(beta),cos(beta),0],[0,0,1]])*np.mat([[1,0,0],[0,cos(alpha),-sin(alpha)],[0,sin(alpha),cos(alpha)]])

    # R = np.mat([[cos( gamma) *cos( alpha)-cos( beta) *sin( gamma) *sin( alpha) , cos( beta) *cos( gamma) *sin( alpha)+sin( gamma) *cos( alpha) , sin( beta) *sin( alpha) ],
    # [-cos( gamma) *sin( alpha)-cos( beta) *sin( gamma) *cos( alpha) , cos( beta) *cos( gamma) *cos( alpha)-sin( gamma) *sin( alpha) , sin( beta) *cos( alpha) ],
    # [sin( beta) *sin( gamma) , -sin( beta) *cos( gamma) , cos( beta)]])

    K1 = np.mat([[R[0,0]**2,R[0,1]**2,R[0,2]**2],
                 [R[1,0]**2,R[1,1]**2,R[1,2]**2],
                 [R[2,0]**2,R[2,1]**2,R[2,2]**2]])

    K2 = np.mat([[R[0,1]*R[0,2],R[0,2]*R[0,0],R[0,0]*R[0,1]],
                 [R[1,1]*R[1,2],R[1,2]*R[1,0],R[1,0]*R[1,1]],
                 [R[2,1]*R[2,2],R[2,2]*R[2,0],R[2,0]*R[2,1]]])

    K3 = np.mat([[R[1,0]*R[2,0],R[1,1]*R[2,1],R[1,2]*R[2,2]],
                 [R[2,0]*R[0,0],R[2,1]*R[0,1],R[2,2]*R[0,2]],
                 [R[0,0]*R[1,0],R[0,1]*R[1,1],R[0,2]*R[1,2]]])

    K4 = np.mat([
    [R[1,1]*R[2,2]+R[1,2]*R[2,1],R[1,0]*R[2,2]+R[1,2]*R[2,0],R[1,1]*R[2,0]+R[1,0]*R[2,1]],
    [R[0,1]*R[2,2]+R[0,2]*R[2,1],R[0,2]*R[2,0]+R[0,0]*R[2,2],R[0,0]*R[2,1]+R[0,1]*R[2,0]],
    [R[0,1]*R[1,2]+R[0,2]*R[1,1],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[0,0]*R[1,1]+R[0,1]*R[1,0]],
    ])
    K = np.vstack((np.hstack((K1,2*K2)),np.hstack((K3,K4))))



    T_s = np.mat([ [R[0,0]**2,R[1,0]**2,R[2,0]**2,2*R[0,0]*R[1,0],2*R[1,0]*R[2,0],2*R[2,0]*R[0,0]],
            [R[0,1]**2,R[1,1]**2,R[2,1]**2,2*R[0,1]*R[1,1],2*R[1,1]*R[2,1],2*R[2,1]*R[0,1]],
            [R[0,2]**2,R[1,2]**2,R[2,2]**2,2*R[0,2]*R[1,2],2*R[1,2]*R[2,2],2*R[2,2]*R[0,2]],
            [R[0,0]*R[0,1],R[1,0]*R[1,1],R[2,0]*R[2,1],R[0,0]*R[1,1]+R[0,1]*R[1,0],R[1,0]*R[2,1]+R[2,0]*R[1,1],R[2,0]*R[0,1]+R[2,1]*R[0,0]],
            [R[0,1]*R[0,2],R[1,1]*R[1,2],R[2,2]*R[2,1],R[1,2]*R[0,1]+R[0,2]*R[1,1],R[1,1]*R[2,2]+R[2,1]*R[1,2],R[2,1]*R[0,2]+R[0,1]*R[2,2]],
            [R[0,0]*R[0,2],R[1,2]*R[1,0],R[2,2]*R[2,0],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[1,2]*R[2,0]+R[1,0]*R[2,2],R[2,2]*R[0,0]+R[2,0]*R[0,2]]])
    T_e = np.mat([ [R[0,0]**2,R[1,0]**2,R[2,0]**2,R[0,0]*R[1,0],R[1,0]*R[2,0],R[2,0]*R[0,0]],
            [R[0,1]**2,R[1,1]**2,R[2,1]**2,R[0,1]*R[1,1],R[1,1]*R[2,1],R[2,1]*R[0,1]],
            [R[0,2]**2,R[1,2]**2,R[2,2]**2,R[0,2]*R[1,2],R[1,2]*R[2,2],R[2,2]*R[0,2]],
            [2*R[0,0]*R[0,1],2*R[1,0]*R[1,1],2*R[2,0]*R[2,1],R[0,0]*R[1,1]+R[0,1]*R[1,0],R[1,0]*R[2,1]+R[2,0]*R[1,1],R[2,0]*R[0,1]+R[2,1]*R[0,0]],
            [2*R[0,1]*R[0,2],2*R[1,1]*R[1,2],2*R[2,2]*R[2,1],R[1,2]*R[0,1]+R[0,2]*R[1,1],R[1,1]*R[2,2]+R[2,1]*R[1,2],R[2,1]*R[0,2]+R[0,1]*R[2,2]],
            [2*R[0,0]*R[0,2],2*R[1,2]*R[1,0],2*R[2,2]*R[2,0],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[1,2]*R[2,0]+R[1,0]*R[2,2],R[2,2]*R[0,0]+R[2,0]*R[0,2]]])

    # C_EBSP0 = T_s*CH*T_e
    C_EBSP0 = K*CH*K.T
    # print(C_EBSP0)
    return C_EBSP0

def Symmetry_CH(CH, normal_vector):

    R = np.array([[1-2*normal_vector[0]**2,-2*normal_vector[0]*normal_vector[1],-2*normal_vector[0]*normal_vector[2]],[-2*normal_vector[0]*normal_vector[1],1-2*normal_vector[1]**2,-2*normal_vector[1]*normal_vector[2]],[-2*normal_vector[0]*normal_vector[2],-2*normal_vector[1]*normal_vector[2],1-2*normal_vector[2]**2]])

    K1 = np.mat([[R[0,0]**2,R[0,1]**2,R[0,2]**2],
                 [R[1,0]**2,R[1,1]**2,R[1,2]**2],
                 [R[2,0]**2,R[2,1]**2,R[2,2]**2]])

    K2 = np.mat([[R[0,1]*R[0,2],R[0,2]*R[0,0],R[0,0]*R[0,1]],
                 [R[1,1]*R[1,2],R[1,2]*R[1,0],R[1,0]*R[1,1]],
                 [R[2,1]*R[2,2],R[2,2]*R[2,0],R[2,0]*R[2,1]]])

    K3 = np.mat([[R[1,0]*R[2,0],R[1,1]*R[2,1],R[1,2]*R[2,2]],
                 [R[2,0]*R[0,0],R[2,1]*R[0,1],R[2,2]*R[0,2]],
                 [R[0,0]*R[1,0],R[0,1]*R[1,1],R[0,2]*R[1,2]]])

    K4 = np.mat([
    [R[1,1]*R[2,2]+R[1,2]*R[2,1],R[1,0]*R[2,2]+R[1,2]*R[2,0],R[1,1]*R[2,0]+R[1,0]*R[2,1]],
    [R[0,1]*R[2,2]+R[0,2]*R[2,1],R[0,2]*R[2,0]+R[0,0]*R[2,2],R[0,0]*R[2,1]+R[0,1]*R[2,0]],
    [R[0,1]*R[1,2]+R[0,2]*R[1,1],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[0,0]*R[1,1]+R[0,1]*R[1,0]],
    ])
    K = np.vstack((np.hstack((K1,2*K2)),np.hstack((K3,K4))))

    T_s = np.mat([ [R[0,0]**2,R[1,0]**2,R[2,0]**2,2*R[0,0]*R[1,0],2*R[1,0]*R[2,0],2*R[2,0]*R[0,0]],
            [R[0,1]**2,R[1,1]**2,R[2,1]**2,2*R[0,1]*R[1,1],2*R[1,1]*R[2,1],2*R[2,1]*R[0,1]],
            [R[0,2]**2,R[1,2]**2,R[2,2]**2,2*R[0,2]*R[1,2],2*R[1,2]*R[2,2],2*R[2,2]*R[0,2]],
            [R[0,0]*R[0,1],R[1,0]*R[1,1],R[2,0]*R[2,1],R[0,0]*R[1,1]+R[0,1]*R[1,0],R[1,0]*R[2,1]+R[2,0]*R[1,1],R[2,0]*R[0,1]+R[2,1]*R[0,0]],
            [R[0,1]*R[0,2],R[1,1]*R[1,2],R[2,2]*R[2,1],R[1,2]*R[0,1]+R[0,2]*R[1,1],R[1,1]*R[2,2]+R[2,1]*R[1,2],R[2,1]*R[0,2]+R[0,1]*R[2,2]],
            [R[0,0]*R[0,2],R[1,2]*R[1,0],R[2,2]*R[2,0],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[1,2]*R[2,0]+R[1,0]*R[2,2],R[2,2]*R[0,0]+R[2,0]*R[0,2]]])
    T_e = np.mat([ [R[0,0]**2,R[1,0]**2,R[2,0]**2,R[0,0]*R[1,0],R[1,0]*R[2,0],R[2,0]*R[0,0]],
            [R[0,1]**2,R[1,1]**2,R[2,1]**2,R[0,1]*R[1,1],R[1,1]*R[2,1],R[2,1]*R[0,1]],
            [R[0,2]**2,R[1,2]**2,R[2,2]**2,R[0,2]*R[1,2],R[1,2]*R[2,2],R[2,2]*R[0,2]],
            [2*R[0,0]*R[0,1],2*R[1,0]*R[1,1],2*R[2,0]*R[2,1],R[0,0]*R[1,1]+R[0,1]*R[1,0],R[1,0]*R[2,1]+R[2,0]*R[1,1],R[2,0]*R[0,1]+R[2,1]*R[0,0]],
            [2*R[0,1]*R[0,2],2*R[1,1]*R[1,2],2*R[2,2]*R[2,1],R[1,2]*R[0,1]+R[0,2]*R[1,1],R[1,1]*R[2,2]+R[2,1]*R[1,2],R[2,1]*R[0,2]+R[0,1]*R[2,2]],
            [2*R[0,0]*R[0,2],2*R[1,2]*R[1,0],2*R[2,2]*R[2,0],R[0,2]*R[1,0]+R[0,0]*R[1,2],R[1,2]*R[2,0]+R[1,0]*R[2,2],R[2,2]*R[0,0]+R[2,0]*R[0,2]]])

    # C_EBSP0 = T_s*CH*T_e
    C_EBSP0 = K*CH*K.T
    # print(C_EBSP0)
    return C_EBSP0

def calc_Young(CH,npoints = 20):
    elas = Elastic(np.array(CH))
    young = []
    for i in np.linspace(0, 2*np.pi, npoints):
        for j in np.linspace(0, np.pi, npoints//2):
            young.append(elas.Young([i, j]))
    return young

def calc_Shear(CH,npoints = 20):
    elas = Elastic(CH)
    shear = []
    for i in np.linspace(0, np.pi, npoints//2):
        for j in np.linspace(0, 2*np.pi, npoints):
            shear.append(elas.shear3D(i, j)[1])
    return shear

def calc_nu(CH,npoints = 20):
    elas = Elastic(CH)
    nu = []
    for i in np.linspace(0, np.pi, npoints//2):
        for j in np.linspace(0, 2*np.pi, npoints):
            nu.append(elas.poisson3D(i, j)[1])
    return nu

def calc_Kube_anisotropy(CH):
    SH = np.linalg.inv(CH)
    kv = 1/9*(CH[0,0]+CH[1,1]+CH[2,2]+2*(CH[0,1]+CH[0,2]+CH[1,2]))
    Gv = 1/15*(CH[0,0]+CH[1,1]+CH[2,2]-(CH[0,1]+CH[0,2]+CH[1,2])+3*(CH[3,3]+CH[4,4]+CH[5,5]))
    kr = 1/(SH[0,0]+SH[1,1]+SH[2,2]+2*(SH[0,1]+SH[0,2]+SH[1,2]))
    Gr = 15/(4*(SH[0,0]+SH[1,1]+SH[2,2])-4*(SH[0,1]+SH[0,2]+SH[1,2])+3*(SH[3,3]+SH[4,4]+SH[5,5]))

    AL = np.sqrt(np.log(kv/kr)**2+5*np.log(Gv/Gr)**2)
    return AL

def save_data(x, nelx = None,nely = None, nelz = None ,stl_filename = None ,output_filename='volfrac'):
    if nelx is None:
        nelx = x.shape[2]
        nely = x.shape[1]
        nelz = x.shape[0]
    if stl_filename is None:
        stl_filename = create_cube(nelx,nely,nelz)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filename)
    reader.Update()
    poly_data = reader.GetOutput()
    cell_dims = [nelx, nely, nelz]#x, y, z
    #x_min:0 x_max:1, y_min:2,y_max:3,z_min:4,z_max:5
    bounds = poly_data.GetBounds()
    mesh_pitch = [(bounds[1] - bounds[0])/cell_dims[0],
                (bounds[3] - bounds[2])/cell_dims[1],
                (bounds[5] - bounds[4])/cell_dims[2]]
    mins = [bounds[0], bounds[2], bounds[4]]
    points = vtk.vtkPoints()
    for iz in range(cell_dims[2]+1):
        for iy in range(cell_dims[1]+1):
            for ix in range(cell_dims[0]+1):
                xx = ix * mesh_pitch[0] + mins[0]
                xy = iy * mesh_pitch[1] + mins[1]
                xz = iz * mesh_pitch[2] + mins[2]
                # print(xx,xy,xz)
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
    for idx in range(nelx*nely*nelz):
        if x[idx//(nelx*nely),idx//nelx%nely,idx%nelx] !=0:
            cell_list.InsertNextId(idx)
        x_volfrac.InsertNextValue(x[idx//(nelx*nely),idx//nelx%nely,idx%nelx])


    base_mesh.GetCellData().SetScalars(x_volfrac)
    extract_cells = vtk.vtkExtractCells()
    extract_cells.SetInputData(base_mesh)
    extract_cells.SetCellList(cell_list)
    extract_cells.Update()

    if not os.path.exists("out"):
        os.mkdir("out")
    writer = vtk.vtkXMLDataSetWriter()
    vtk_filename = "./out/"+output_filename+".vtu"
    writer.SetFileName(vtk_filename)
    writer.SetInputData(extract_cells.GetOutput())
    writer.Update()

    return vtk_filename

def create_cube(len_x,len_y,len_z):

    coordinate=[[0,0,0], [len_x,0,0], [len_x,len_y,0], [0,len_y,0], [0,0,len_z], [len_x,0,len_z], [len_x,len_y,len_z],[0,len_y,len_z]]

    # 6 face of the cube
    face = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()

    for i in range(8):
        points.InsertPoint(i, coordinate[i])

    for i in range(6):
        polys.InsertNextCell(4, face[i])

    #绑定标量数据
    for i in range(8):
        scalars.InsertTuple1(i, i)

    #关联数据
    cube.SetPoints(points)
    cube.SetPolys(polys)
    cube.GetPointData().SetScalars(scalars)
    writer = vtk.vtkSTLWriter()
    output_filename = "out/cube%i%i%i.stl"%(len_x,len_y,len_z)
    writer.SetFileName(output_filename)
    writer.SetInputData(cube)
    writer.Update()
    writer.Write()

    return output_filename

def get_x(mesh_size, rod, stl_file, precision = 100,center_point = (5,5,5,),n_index = -1):
    if rod is None:
        x,_ = micro.read_stl(mesh_size,stl_file,0,1,False,5)
        return x
    else:
        x_all = np.zeros((mesh_size,mesh_size,mesh_size))
        for r_index in range(11):
            if rod[r_index] <0.2:
                continue
            if n_index !=-1:
                output_filename = line2mesh('Cubic-BCC_vtk/%d_%d_%.1f %.1f %.1f.vtk'%(r_index+1,n_index,center_point[0],center_point[1],center_point[2]),rod[r_index],precision)
            else :
                output_filename = line2mesh('Cubic-BCC_vtk/%d_%.1f %.1f %.1f.vtk'%(r_index+1,center_point[0],center_point[1],center_point[2]),rod[r_index],precision)
            x,_ = micro.read_stl(mesh_size,output_filename,1,1,False,8)

            if os.path.isfile(output_filename):
                os.remove(output_filename)

            x_all = x_all + x
            x_all[np.where(x_all>1)] = 1
        # rf.save_data(x_all,mesh_size,mesh_size,mesh_size,stl_filename)
        return x_all

def get_x_from_VTK(mesh_size, rod, vtk_filenames):
    x_all = np.zeros((mesh_size,mesh_size,mesh_size))
    for index, vtk_filename in enumerate(vtk_filenames):
        x,_ = VTK2Mesh(mesh_size,1,vtk_filename,rod[index])
        x_all = x_all + x
        x_all[np.where(x_all>1)] = 1
    return x_all

def get_stl(mesh_size, rod, stl_file, precision = 100,center_point = (5,5,5,)):
    if rod is None:
        return
    else:
        for r_index in range(11):
            if rod[r_index] <0.2:
                continue
            output_filename = line2mesh('Cubic-BCC_vtk/%d_%.1f %.1f %.1f.vtk'%(r_index+1,center_point[0],center_point[1],center_point[2]),rod[r_index],precision)
        return

def calc_MaxYoungs_dirction(CH):
    elas = Elastic(np.array(CH))
    young = []
    theta = []
    phi = []
    for i in np.linspace(0, 2*np.pi, 200): #theta
        for j in np.linspace(0, np.pi, 200//2): #phi
            theta.append(i)
            phi.append(j)
            young.append(elas.Young([i, j]))
    max_theta = theta[young.index(max(young))]
    max_phi = phi[young.index(max(young))]
    E_direct = np.array([np.sin(max_theta)*np.cos(max_phi),np.sin(max_theta)*np.sin(max_phi),np.cos(max_theta)])
    return E_direct

def line2mesh(filename,rod_diameter,r1=0,r2=0,precision = 100, output_filename = ""):
    if not os.path.exists(filename):
        print(filename+' is not exist.')
        sys.exit()
    reader = vtkPolyDataReader()
    reader.SetFileName(filename)

    imp = vtkImplicitModeller()
    imp.SetInputConnection(reader.GetOutputPort())
    imp.SetSampleDimensions(precision, precision, precision)
    # imp.AdjustBoundsOff()
    imp.SetMaximumDistance(0.5)
    imp.SetModelBounds(0, 10,0, 10,0, 10)
    imp.SetAdjustDistance(0.0001)
    imp.SetNumberOfThreads (100)

    contour = vtkContourFilter()
    contour.SetInputConnection(imp.GetOutputPort())
    contour.SetValue(0, rod_diameter) #SetValue (int i, double value)
    polydatanormals = vtk.vtkPolyDataNormals()
    polydatanormals.SetInputConnection(contour.GetOutputPort())
    polydatanormals.SetFlipNormals(True)
    polydatanormals.Update()

    # impMapper = vtkPolyDataMapper()
    # impMapper.SetInputConnection(contour.GetOutputPort())
    # impMapper.ScalarVisibilityOff()

    writer = vtk.vtkSTLWriter()
    if output_filename == "":
        output_filename = "Cubic-BCC_vtk/"+Path(filename).stem+"_%.1f %.1f.stl"%(r1,r2)

    # output_filename = filename.split(".")[0]+"_%.1f%.1f.stl"%(r1,r2)
    writer.SetFileName(output_filename)
    writer.SetInputConnection(polydatanormals.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    return output_filename

def line2mesh_display(filename,rod_diameter,r1=0,r2=0,precision = 50, output_filename = ""):
    if not os.path.exists(filename):
        print(filename+' is not exist.')
        sys.exit()
    reader = vtkPolyDataReader()
    reader.SetFileName(filename)

    imp = vtkImplicitModeller()
    imp.SetInputConnection(reader.GetOutputPort())
    imp.SetSampleDimensions(precision, precision, precision)
    # imp.AdjustBoundsOff()
    imp.SetMaximumDistance(0.5)
    imp.SetModelBounds(0, 10,0, 10,0, 10)
    imp.SetAdjustDistance(0.0001)
    imp.SetNumberOfThreads (100)

    contour = vtkContourFilter()
    contour.SetInputConnection(imp.GetOutputPort())
    contour.SetValue(0, rod_diameter) #SetValue (int i, double value)
    polydatanormals = vtk.vtkPolyDataNormals()
    polydatanormals.SetInputConnection(contour.GetOutputPort())
    polydatanormals.SetFlipNormals(True)
    polydatanormals.Update()

    decimationFilter = vtk.vtkDecimatePro()
    decimationFilter.SetInputData(polydatanormals.GetOutput())
    decimationFilter.SetTargetReduction(0.01) #Set#Specify the desired reduction in the total number of polygons
    decimationFilter.SetBoundaryVertexDeletion(True) #Turn on/off the deletion of vertices on the boundary of a mesh
    decimationFilter.PreserveTopologyOn()
    decimationFilter.Update()

    writer = vtk.vtkSTLWriter()
    if output_filename == "":
        output_filename = "Cubic-BCC_vtk/"+Path(filename).stem+"_%.1f %.1f.stl"%(r1,r2)

    # output_filename = filename.split(".")[0]+"_%.1f%.1f.stl"%(r1,r2)
    writer.SetFileName(output_filename)
    writer.SetInputConnection(decimationFilter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    return output_filename

def create_vtk(center_point,index = -1):
    return_filename = []
    for i in range(11):
        file = 'Cubic-BCC_vtk/'+str(i+1)+'.vtk'
        with open(file) as f:
            data = f.readlines()
        data[14] = str(center_point[0])+" " +str(center_point[1])+" "+str(center_point[2])+"\n"
        if index == -1:
            write_file = 'Cubic-BCC_vtk/%d_%.1f %.1f %.1f.vtk'%(i+1,center_point[0],center_point[1],center_point[2])
        else:
            write_file = 'Cubic-BCC_vtk/%d_%d_%.1f %.1f %.1f.vtk'%(i+1,index,center_point[0],center_point[1],center_point[2])
        return_filename.append(write_file)
        with open(write_file, mode='w', encoding='utf-8') as file_obj:
            file_obj.writelines(data)
        # print(data)
    return return_filename

def stress_visualization(stress, vtp_filename, N=7):
    node = np.array( [
        [ 1.0, 0.0, 0.0], # 0
        [-1.0, 0.0, 0.0], # 1
        [ 0.0, 1.0, 0.0], # 2
        [ 0.0,-1.0, 0.0], # 3
        [ 0.0, 0.0, 1.0], # 4
        [ 0.0, 0.0,-1.0]  # 5
        ] )

    octahedron = np.array( [
        [ 0, 2, 4 ],
        [ 2, 1, 4 ],
        [ 1, 3, 4 ],
        [ 3, 0, 4 ],
        [ 0, 5, 2 ],
        [ 2, 5, 1 ],
        [ 1, 5, 3 ],
        [ 3, 5, 0 ],
        ] )

    nodeChild = {}
    for i in range(len(node)):
        nodeChild['{clone}+{clone}'.format(clone=str(i))] = i

    connectivity = np.vstack([SierpinskySpherical(t,N,nodeChild,node) for t in octahedron])


    stress_size, stress_vector = np.linalg.eig(stress)
    stress_size_sort = np.argsort(np.abs(stress_size))[::-1]
    stress_vector = stress_vector.T
    stress_vector = stress_vector[stress_size_sort]
    stress_vector = stress_vector.T
    if np.linalg.det(stress_vector)<0:
        stress_vector = - stress_vector
    stress_size = stress_size[stress_size_sort]
    if np.abs(np.arccos(np.cross(stress_vector[:,0],stress_vector[:,1]).dot(stress_vector[:,2]))) > 0.1:
        stress_vector[:,2] = - stress_vector[:,2]

    for i,n in enumerate(node):
        direct = np.linalg.inv(stress_vector).dot(n)
        # theta = np.arccos(n[2])
        # phi = np.angle(complex(n[0], n[1]))
        theta = np.arccos(direct[2])
        phi = np.angle(complex(direct[0], direct[1]))
        dir_x = np.sin(theta)*np.cos(phi)/stress_size[0]
        dir_y = np.sin(theta)*np.sin(phi)/(0.5*stress_size[0])
        dir_z = np.cos(theta)/(0.5*stress_size[0])
        r = 1/np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)

        # node[i] *= np.abs(np.sum(stress_size*direct**2))
        # print(np.sum(stress_size*direct**2)-n.dot(stress.dot(np.reshape(n,(3,1)))))
        # node[i] *= n.dot(stress.dot(np.reshape(n,(3,1))))
        node[i] *= r # n方向的大小
        # print(n,r)
    vtk_writeData(vtp_filename,node,connectivity)

'''
\sigma_n = \sigma_1*l^2+\sigma_2*m^2+\sigma_3*n^2
\tau_n = \sigma_1^2*l^2+\sigma_2^2*m^2+\sigma_3^2*n^2-\sigma_n^2
where, \sigma_1 is the first principal stress, (l,m,n) direct vector in x'y'z' Coordinate.
'''
def calc_Stress(stress, npoints =20):

    stress_size, stress_vector = np.linalg.eig(stress)
    stress_size_sort = np.argsort(np.abs(stress_size))[::-1]
    stress_vector = stress_vector.T
    stress_vector = stress_vector[stress_size_sort]
    stress_vector = stress_vector.T
    if np.linalg.det(stress_vector)<0:
        stress_vector = - stress_vector
    stress_size = stress_size[stress_size_sort]
    if np.abs(np.arccos(np.cross(stress_vector[:,0],stress_vector[:,1]).dot(stress_vector[:,2]))) > 0.1:
        stress_vector[:,2] = - stress_vector[:,2]

    stress_list = []

    for i in np.linspace(0, np.pi, npoints//2):
        for j in np.linspace(0, 2*np.pi, npoints):
            n = np.array([np.sin(i)*np.cos(j),np.sin(i)*np.sin(j),np.cos(i)])
            # n = n/np.linalg.norm(n) #norm

            direct = np.linalg.inv(stress_vector).dot(n)
            # theta = np.arccos(n[2])
            # phi = np.angle(complex(n[0], n[1]))
            theta = np.arccos(direct[2])
            phi = np.angle(complex(direct[0], direct[1]))
            dir_x = np.sin(theta)*np.cos(phi)/stress_size[0]
            dir_y = np.sin(theta)*np.sin(phi)/(0.5*stress_size[0])
            dir_z = np.cos(theta)/(0.5*stress_size[0])
            r = 1/np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            stress_list.append(r)
    return stress_list

def calc_max_stress(stress_list, npoints = 20):
    index = np.argmax(stress_list)
    theta = np.linspace(0, np.pi, npoints//2)[index//npoints]
    phi = np.linspace(0, 2*np.pi, npoints)[index % npoints]
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def TPMS2Mesh(mesh_size, small_mesh_size, func = lambda x,y,z: np.cos(2*np.pi*x)+np.cos(2*np.pi*y)+np.cos(2*np.pi*z), const = 0.7, Rotation = [[1,0,0],[0,1,0],[0,0,1]]):
    x_value = np.zeros((mesh_size, mesh_size, mesh_size))
    x_value_small = np.zeros((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)
    sdf_distance = np.zeros((mesh_size, mesh_size, mesh_size))

    # for elz in range(mesh_size):
    #     for elx in range(mesh_size):
    #         for ely in range(mesh_size)[::-1]:
    #             mesh_pitch = 1/mesh_size
    #             center_point = ((elx+0.5)*mesh_pitch,(ely+0.5)*mesh_pitch,(elz+0.5)*mesh_pitch)
    #             value = func(center_point[0], center_point[1], center_point[2])
    #             if value <= const:
    #                 x_value[elz,ely,elx] = 1
    #             else :
    #                 x_value[elz,ely,elx] = 0

    mesh_pitch = 1/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = mesh_pitch/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    bias_small = mesh_dot * small_mesh_pitch

    center_points_small = np.repeat(center_points, small_mesh_size**3, axis = 0)
    center_points_small = center_points_small + np.tile(bias_small,(mesh_size**3,1))
    xyz_center_points_small = (center_points_small-0.5).dot(np.linalg.inv(np.array(Rotation))) + 0.5
    if type(func) != list:
        value_small = func(xyz_center_points_small[:,0], xyz_center_points_small[:,1], xyz_center_points_small[:,2])
    else:
        for func_s in func:
            value_small = np.zeros((mesh_size**3*small_mesh_size**3,))
            value_small = value_small + func_s(xyz_center_points_small[:,0], xyz_center_points_small[:,1], xyz_center_points_small[:,2])
    x_value_small[np.where(value_small < const)] = 1
    for index in range(mesh_size**3):
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(x_value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
        sdf_distance[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
    return x_value, sdf_distance

def Spinodoid2Mesh(mesh_size, small_mesh_size, rou, theta1, theta2, theta3, e1 = (1,0,0), e2 = (0,1,0), e3 = (0,0,1), N = 100, beta = 20):

    x_value = np.zeros((mesh_size, mesh_size, mesh_size))
    x_value_small = np.zeros((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)
    sdf_distance = np.zeros((mesh_size, mesh_size, mesh_size))

    mesh_pitch = 1/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = mesh_pitch/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    bias_small = mesh_dot * small_mesh_pitch

    center_points_small = np.repeat(center_points, small_mesh_size**3, axis = 0)
    center_points_small = center_points_small + np.tile(bias_small,(mesh_size**3,1))
    center_points_small = center_points_small - 0.5

    # np.random.seed(0)
    # theta = np.arccos(np.arange(-1, 1, 2/N))
    # theta = np.append(theta, 0)
    # phi = np.arange(0, 2*np.pi, 2*np.pi/N)

    # [theta_grid, phi_grid] = np.meshgrid(theta, phi)
    # theta_grid = theta_grid.reshape((-1))
    # phi_grid = phi_grid.reshape((-1))
    # vectors = np.column_stack((np.sin(theta_grid)*np.cos(phi_grid),np.sin(theta_grid)*np.sin(phi_grid),np.cos(theta_grid)))
    np.random.seed(N**2)
    vectors = np.random.randn(N**2, 3)
    bias = np.random.uniform(0, 2*np.pi, N**2)

    vectors = vectors/np.linalg.norm(vectors, axis = 1).reshape((-1,1))
    vectors = np.vstack((vectors.dot(np.vstack((e1,e2,e3))),vectors.dot(np.vstack((e2,e3,e1))),vectors.dot(np.vstack((e3,e1,e2)))))
    bias = np.hstack((bias,bias,bias))
    k = vectors[np.where((np.abs(np.sum(vectors*e1,axis=1)) > np.cos(theta1)) | (np.abs(np.sum(vectors*e2,axis=1)) > np.cos(theta2)) | (np.abs(np.sum(vectors*e3,axis=1)) > np.cos(theta3)))]
    bias= bias[np.where((np.abs(np.sum(vectors*e1,axis=1)) > np.cos(theta1)) | (np.abs(np.sum(vectors*e2,axis=1)) > np.cos(theta2)) | (np.abs(np.sum(vectors*e3,axis=1)) > np.cos(theta3)))]
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(k[:,0], k[:,1], k[:,2], c='b', marker='o', label='Data Points')
    # plt.show()

    N = k.shape[0]
    np.random.seed(N)
    # bias = np.random.uniform(0, 2*np.pi, N)
    beta = beta
    value_small = np.sqrt(2/N)*np.sum(np.cos(beta * (np.kron(center_points_small[:,0],k[:,0]) + np.kron(center_points_small[:,1],k[:,1]) + np.kron(center_points_small[:,2],k[:,2])) + np.kron(np.ones((center_points_small.shape[0])),bias)).reshape((-1,N)),axis = 1)

    x_value_small[np.where(value_small < scipy.special.erfinv(2 * rou - 1))] = 1
    for index in range(mesh_size**3):
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(x_value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
        sdf_distance[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
    return x_value, sdf_distance

def Spinodoid2MeshV2(mesh_size, small_mesh_size, rou, theta1, theta2, theta3, e1 = (1,0,0), e2 = (0,1,0), e3 = (0,0,1), N = 100, beta = 20):

    x_value = np.zeros((mesh_size, mesh_size, mesh_size))
    x_value_small = np.zeros((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)
    sdf_distance = np.zeros((mesh_size, mesh_size, mesh_size))

    mesh_pitch = 1/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index  bound[0,1]
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = mesh_pitch/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    bias_small = mesh_dot * small_mesh_pitch

    center_points_small = np.repeat(center_points, small_mesh_size**3, axis = 0)
    center_points_small = center_points_small + np.tile(bias_small,(mesh_size**3,1))
    center_points_small = center_points_small - 0.5


    #斐波那契格点构造法
    phi = (1 + np.sqrt(5)) / 2
    xx = (np.arange(1, N**2 + 1) / phi) % 1
    xy = np.arange(1, N**2 + 1) / N**2
    theta = 2 * np.pi * xx
    r = np.sqrt(xy)
    phi = 2 * np.arcsin(r)  # latitude
    xx = np.cos(theta) * np.sin(phi)
    xy = np.sin(theta) * np.sin(phi)
    xz = np.cos(phi)
    vectors = np.hstack((xx.reshape((-1,1)),xy.reshape((-1,1)),xz.reshape((-1,1))))
    # vectors = np.random.randn(N**2, 3)
    # vectors = vectors/np.linalg.norm(vectors, axis = 1).reshape((-1,1))


    vectors = np.vstack((vectors.dot(np.vstack((e1,e2,e3))),vectors.dot(np.vstack((e2,e3,e1))),vectors.dot(np.vstack((e3,e1,e2)))))
    np.random.seed(N)
    bias = np.random.uniform(0, 2*np.pi, N**2)
    bias = np.hstack((bias,bias,bias))
    k = vectors[np.where((np.abs(np.sum(vectors*e1,axis=1)) > np.cos(theta1)) | (np.abs(np.sum(vectors*e2,axis=1)) > np.cos(theta2)) | (np.abs(np.sum(vectors*e3,axis=1)) > np.cos(theta3)))]
    bias= bias[np.where((np.abs(np.sum(vectors*e1,axis=1)) > np.cos(theta1)) | (np.abs(np.sum(vectors*e2,axis=1)) > np.cos(theta2)) | (np.abs(np.sum(vectors*e3,axis=1)) > np.cos(theta3)))]

    # * 检查点
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(k[:,0], k[:,1], k[:,2], c='b', marker='o', label='Data Points')
    # plt.show()

    N = k.shape[0]
    beta = beta
    value_small = np.sqrt(2/N)*np.sum(np.cos(beta * (np.kron(center_points_small[:,0],k[:,0]) + np.kron(center_points_small[:,1],k[:,1]) + np.kron(center_points_small[:,2],k[:,2])) + np.kron(np.ones((center_points_small.shape[0])),bias)).reshape((-1,N)),axis = 1)

    x_value_small[np.where(value_small < scipy.special.erfinv(2 * rou - 1))] = 1
    for index in range(mesh_size**3):
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(x_value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
        sdf_distance[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
    return x_value, sdf_distance


def TPMS_like2Mesh(mesh_size, small_mesh_size, rou, weight=(1,1,1), e1 = (1,0,0), e2 = (0,1,0), e3 = (0,0,1), beta = 10*np.pi):

    x_value = np.zeros((mesh_size, mesh_size, mesh_size))
    x_value_small = np.zeros((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)
    sdf_distance = np.zeros((mesh_size, mesh_size, mesh_size))

    mesh_pitch = 1/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index  bound[0,1]
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))

    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = mesh_pitch/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    bias_small = mesh_dot * small_mesh_pitch

    center_points_small = np.repeat(center_points, small_mesh_size**3, axis = 0)
    center_points_small = center_points_small + np.tile(bias_small,(mesh_size**3,1))
    # center_points_small = center_points_small - 0.5

    k = np.array((e1,e2,e3))
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(k[:,0], k[:,1], k[:,2], c='b', marker='o', label='Data Points')
    # plt.show()

    value_small = np.sqrt(2)/np.linalg.norm(weight)*np.sum(np.cos(beta * (np.kron(center_points_small[:,0],k[:,0]) + np.kron(center_points_small[:,1],k[:,1]) + np.kron(center_points_small[:,2],k[:,2]))).reshape((-1,3)),axis = 1)

    x_value_small[np.where(value_small < scipy.special.erfinv(2 * rou - 1))] = 1
    for index in range(mesh_size**3):
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(x_value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
        sdf_distance[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
    return x_value, sdf_distance

def voxel_visualization(V):
    from matplotlib import cm
    # 获取体素矩阵的最小值和最大值
    vmin = V.min()
    vmax = V.max()

    # 创建一个归一化器，将体素矩阵值映射到[0, 1]范围
    norm = plt.Normalize(vmin, vmax)

    # 使用归一化器将体素矩阵值映射到颜色
    # https://zhuanlan.zhihu.com/p/158871093
    # colors = cm.coolwarm(norm(V))
    colors = cm.viridis(norm(V))
    # colors = cm.Oranges(norm(V))
    # colors = cm.Paired(norm(V))  # 离散色阶

    # 创建一个3D绘图
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # 设置相机角度
    elevation_angle = 30  # 仰角（垂直角度），可以根据需要调整
    azimuthal_angle = 45-90 # 方位角（水平角度），可以根据需要调整
    ax.view_init(elevation_angle, azimuthal_angle)

    # 使用ax.voxels绘制体素矩阵
    ax.voxels(V, facecolors=colors, edgecolor='k',linewidth=0.5)
    # ax.voxels(V, facecolors=colors)
    # ax.set_axis_off()
    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # 设置标题
    # ax.set_title('Voxel Matrix Visualization')
    plt.savefig('out/voxel_visualization.png', dpi=1200, bbox_inches='tight')
    # plt.show()

def voxel_visualization2(V):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 假设体素矩阵V已经给出，大小为20x20x20
    # V中的值应介于0和1之间
    # V = ...

    # 定义正方体的顶点和面
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])

    faces = np.array([
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [4, 5, 6, 7]
    ])

    # 创建一个3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 遍历体素矩阵中的所有体素
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                # 如果体素值大于0，添加一个正六面体并根据体素值设置颜色
                if V[x, y, z] > 0:
                    cube_vertices = vertices + np.array([x - 1, y - 1, z - 1])
                    color_index = int(np.round(V[x, y, z] * 255))
                    face_colors = plt.cm.viridis(color_index / 255)[:3]
                    poly = Poly3DCollection(cube_vertices[faces], facecolors=face_colors, edgecolors='k', linewidths=0.5)
                    ax.add_collection3d(poly)

    # 设置视角
    ax.view_init(30, 45)

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # 显示颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
    cbar.set_label('Voxel value')

    plt.show()

def VTK2Mesh_old(mesh_size, vtk_filename, rod_radius):

    # 支持定义点和线段的vtk文件
    point_list = []
    line_list =[]
    x_value = np.zeros((mesh_size, mesh_size, mesh_size))

    with open(vtk_filename) as f:
        data = f.readlines()
    for index in range(len(data)):
        if 'POINTS' in data[index]:
            point_number = int(data[index].split(' ')[1])
            for index2 in range(point_number):
                point_list.append(np.array([float(data[index+index2+1].split(" ")[i]) for i in range(3)]))
        if 'LINES' in data[index]:
            line_number = int(data[index].split(' ')[1])
            for index2 in range(line_number):
                line_list.append([int(data[index+index2+1].split(" ")[i]) for i in range(1,3)])

    mesh_pitch = 10/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))


    for line in line_list:
        v = center_points - point_list[line[0]]
        s = point_list[line[1]] - point_list[line[0]] # 方向向量
        d = np.linalg.norm(np.cross(v,s),axis = 1)/np.sqrt(s[0]**2+s[1]**2+s[2]**2)
        t = np.inner(v,s)/(s[0]**2+s[1]**2+s[2]**2) #判断垂足是否处于线段范围

        out_t_index = np.where((t>1) & (t<0))
        d[out_t_index] = [ min(np.linalg.norm(center_points[i]  - point_list[line[0]]), np.linalg.norm(center_points[i] - point_list[line[1]])) for i in out_t_index ]
        index = np.where(d < rod_radius)[0]
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = 1

    return x_value

# 定义三次贝塞尔插值函数
def quadratic_bezier(t, p0, p1, p2):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def r_roughness(t, rod_radius, roughness):
    # 处理表面粗糙度
    num_points = 11
    pitch = 2/(num_points-1)
    r_roughness = rod_radius *np.ones((num_points)) + np.random.uniform(-roughness, roughness, size = num_points)
    slice = np.array(list(map(int,t//pitch)))
    return quadratic_bezier((t%pitch)/pitch,r_roughness[2*slice],r_roughness[2*slice+1],r_roughness[2*slice+2])



def VTK2Mesh(mesh_size,small_mesh_size, vtk_filename, rod_radius, roughness = 0):

    # 支持定义点和线段的vtk文件
    point_list = []
    line_list =[]
    x_value = np.zeros((mesh_size, mesh_size, mesh_size))
    x_value_small = np.zeros((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)
    sdf_distance = np.zeros((mesh_size, mesh_size, mesh_size))
    sdf_distance_small = 100 * np.ones((mesh_size**3 * small_mesh_size**3),dtype=np.uint8)

    with open(vtk_filename) as f:
        data = f.readlines()
    for index in range(len(data)):
        if 'POINTS' in data[index]:
            point_number = int(data[index].split(' ')[1])
            for index2 in range(point_number):
                point_list.append(np.array([float(data[index+index2+1].split(" ")[i]) for i in range(3)]))
        if 'LINES' in data[index]:
            line_number = int(data[index].split(' ')[1])
            for index2 in range(line_number):
                line_list.append([int(data[index+index2+1].split(" ")[i]) for i in range(1,3)])

    mesh_pitch = 10/mesh_size
    center_points_meshgrid = np.meshgrid((np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)+0.5)*mesh_pitch, (np.arange(mesh_size)[::-1]+0.5)*mesh_pitch) #xzy index
    center_points_x = center_points_meshgrid[0].reshape((-1,1))
    center_points_z = center_points_meshgrid[1].reshape((-1,1))
    center_points_y = center_points_meshgrid[2].reshape((-1,1))
    center_points = np.hstack((center_points_x, center_points_y, center_points_z))


    #对边界的单元重新采样并赋予中间值
    small_mesh_pitch = mesh_pitch/small_mesh_size
    dis_list = np.arange(-(small_mesh_size - 1)/2,(small_mesh_size - 1)/2 + 1,1)
    [mesh_dot_x, mesh_dot_y, mesh_dot_z]= np.meshgrid(dis_list, dis_list, dis_list)
    mesh_dot_x = mesh_dot_x.reshape((small_mesh_size**3,1))
    mesh_dot_y = mesh_dot_y.reshape((small_mesh_size**3,1))
    mesh_dot_z = mesh_dot_z.reshape((small_mesh_size**3,1))
    mesh_dot = np.hstack((mesh_dot_x,mesh_dot_y,mesh_dot_z))
    bias_small = mesh_dot * small_mesh_pitch

    center_points_small = np.repeat(center_points, small_mesh_size**3, axis = 0)
    center_points_small = center_points_small + np.tile(bias_small,(mesh_size**3,1))

    for line in line_list:
        v = center_points_small - point_list[line[0]]
        s = point_list[line[1]] - point_list[line[0]] # 方向向量
        d = np.linalg.norm(np.cross(v,s),axis = 1)/np.sqrt(s[0]**2+s[1]**2+s[2]**2)
        t = np.inner(v,s)/(s[0]**2+s[1]**2+s[2]**2) #判断垂足是否处于线段范围
        # Q = point_list[line[0]] + t.reshape((-1,1)) * s  # 垂足点

        out_t_index = np.where((t>1) | (t<0))
        d[out_t_index] = [ min(np.linalg.norm(center_points_small[i]  - point_list[line[0]]), np.linalg.norm(center_points_small[i] - point_list[line[1]])) for i in out_t_index ]
        t[np.where((t>1))] = 1
        t[np.where((t<0))] = 0
        if roughness == 0:
            index = np.where(d < rod_radius)[0]
        else:
            index = np.where(d < r_roughness(t, rod_radius, roughness))[0]

        x_value_small[index] = 1
        sdf_distance_small = np.minimum(sdf_distance_small,d)

    for index in range(mesh_size**3):
        x_value[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(x_value_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
        sdf_distance[index//(mesh_size**2), mesh_size - index%mesh_size - 1, index//mesh_size%mesh_size] = np.average(sdf_distance_small[index*small_mesh_size**3:(index+1)*small_mesh_size**3])
    return x_value, sdf_distance

def is_discrete_point(matrix, i, j, k):
    if matrix[i, j, k] != 0:
        if (
            matrix[i - 1, j, k] == 0
            and matrix[i + 1, j, k] == 0
            and matrix[i, j - 1, k] == 0
            and matrix[i, j + 1, k] == 0
        ):
            return True
    return False

def find_discrete_regions(matrix):
    visited = np.zeros_like(matrix, dtype=bool)
    shape = matrix.shape
    regions = []

    def dfs(i, j, k, region):
        if (
            0 <= i < shape[0]
            and 0 <= j < shape[1]
            and 0 <= k < shape[2]
            and not visited[i, j, k]
            and is_discrete_point(matrix, i, j, k)
        ):
            visited[i, j, k] = True
            region.append((i, j, k))

            for x, y, z in [(i-1, j, k), (i+1, j, k), (i, j-1, k), (i, j+1, k)]:
                dfs(x, y, z, region)

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                if not visited[i, j, k] and is_discrete_point(matrix, i, j, k):
                    region = []
                    dfs(i, j, k, region)
                    regions.append(region)

    return regions

from scipy import ndimage
def remove_largest_connected_region(label_matrix):
    # 使用SciPy的ndimage.label函数找到连通区域
    labeled_array, num_features = ndimage.label(label_matrix)

    # 计算每个连通区域的大小
    sizes = ndimage.sum(np.ones_like(label_matrix), labeled_array, range(1, num_features + 1))

    # 对连通区域进行排序
    sorted_indices = np.argsort(sizes)[::-1]

    # 删除最大的连通区域
    largest_label = sorted_indices[0] + 1
    mask = labeled_array == largest_label

    # 创建一个不包含最大连通区域的新标签矩阵
    filtered_label_matrix = labeled_array.copy()
    filtered_label_matrix[mask] = 0

    return filtered_label_matrix

# 假设您已经获得了一个表示体素矩阵的NumPy数组，例如label_matrix
# 使用remove_largest_connected_region函数处理label_matrix
# filtered_label_matrix = remove_largest_connected_region(label_matrix)

def get_largest_connected_region(label_matrix):
    # 使用SciPy的ndimage.label函数找到连通区域
    labeled_array, num_features = ndimage.label(label_matrix)

    # 计算每个连通区域的大小
    sizes = ndimage.sum(np.ones_like(label_matrix), labeled_array, range(1, num_features + 1))

    # 对连通区域进行排序
    sorted_indices = np.argsort(sizes)[::-1]

    # 删除最大的连通区域
    largest_label = sorted_indices[0] + 1
    mask = labeled_array == largest_label

    # 创建一个不包含最大连通区域的新标签矩阵
    filtered_label_matrix = np.zeros_like(label_matrix)
    filtered_label_matrix[mask] = 1

    return filtered_label_matrix


def get_TPMS_func(label):

    #! [1] Yoo D J. Computer-aided porous scaffold design for tissue engineering using triply periodic minimal surfaces[J]. International Journal of Precision Engineering and Manufacturing, 2011, 12(1): 61-71.
    if label =='Strut P':
        func = lambda x,y,z:np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)
    elif label == 'Strut D':
        func = lambda x,y,z:np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z)
    elif label == 'Strut G':
        func = lambda x,y,z: np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + np.sin(2*np.pi*z) * np.cos(2*np.pi*x) + np.sin(2*np.pi*y) * np.cos(2*np.pi*z)
    elif label == 'Strut I-WP':
        func = lambda x,y,z: 2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))-(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z))
    elif label == 'Strut F-RD':
        func = lambda x,y,z: 4 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - (np.cos(2*2*np.pi*x)*np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*y)*np.cos(2*2*np.pi*z)+np.cos(2*2*np.pi*z)*np.cos(2*2*np.pi*x))
    elif label == 'Strut L':
        func = lambda x,y,z: 0.5 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(z) + np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) * np.sin(2*np.pi*x) + np.sin(2 * 2*np.pi*z) * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) + 0.15
    elif label == 'Strut Tubular P':
        func = lambda x,y,z: 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)) - 5.1 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) + np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*z) * np.cos(2*np.pi*x)) - 14.6
    elif label == 'Strut Tubular G':
        func = lambda x,y,z: 10 * (np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.cos(2*np.pi*z) * np.sin(2*np.pi*x)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) - 14
    elif label == 'Strut I2-Y':
        func = lambda x,y,z: -2 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.sin(2*np.pi*x) * np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2 * 2*np.pi*z)) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*z)

    elif label == 'Strut BCC':
        func = lambda x,y,z: -2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))+(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z))


    elif label =='Sheet P':
        func = lambda x,y,z:(np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z))**2
    elif label == 'Sheet D':
        func = lambda x,y,z:(np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z))**2
    elif label == 'Sheet G':
        func = lambda x,y,z: (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + np.sin(2*np.pi*z) * np.cos(2*np.pi*x) + np.sin(2*np.pi*y) * np.cos(2*np.pi*z))**2
    elif label == 'Sheet I-WP':
        func = lambda x,y,z: (2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))-(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z)))**2
    elif label == 'Sheet F-RD':
        func = lambda x,y,z: (4 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - (np.cos(2*2*np.pi*x)*np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*y)*np.cos(2*2*np.pi*z)+np.cos(2*2*np.pi*z)*np.cos(2*2*np.pi*x)))**2
    elif label == 'Sheet L':
        func = lambda x,y,z: (0.5 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(z) + np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) * np.sin(2*np.pi*x) + np.sin(2 * 2*np.pi*z) * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) + 0.15 )**2
    elif label == 'Sheet Tubular P':
        func = lambda x,y,z: (10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)) - 5.1 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) + np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*z) * np.cos(2*np.pi*x)) - 14.6)**2
    elif label == 'Sheet Tubular G':
        func = lambda x,y,z: (10 * (np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.cos(2*np.pi*z) * np.sin(2*np.pi*x)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) - 14)**2
    elif label == 'Sheet I2-Y':
        func = lambda x,y,z: (-2 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.sin(2*np.pi*x) * np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2 * 2*np.pi*z)) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*z))**2
    return func

def get_TPMS_funcV2(label):

    #! [1] Yoo D J. Computer-aided porous scaffold design for tissue engineering using triply periodic minimal surfaces[J]. International Journal of Precision Engineering and Manufacturing, 2011, 12(1): 61-71.
    if label =='Strut P':
        func = lambda x,y,z:np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)
    elif label == 'Strut D':
        func = lambda x,y,z:np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z)
    elif label == 'Strut G':
        func = lambda x,y,z: np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + np.sin(2*np.pi*z) * np.cos(2*np.pi*x) + np.sin(2*np.pi*y) * np.cos(2*np.pi*z)
    elif label == 'Strut I-WP':
        func = lambda x,y,z: 2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))-(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z))
    elif label == 'Strut F-RD':
        func = lambda x,y,z: 4 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - (np.cos(2*2*np.pi*x)*np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*y)*np.cos(2*2*np.pi*z)+np.cos(2*2*np.pi*z)*np.cos(2*2*np.pi*x))
    elif label == 'Strut L':
        func = lambda x,y,z: 0.5 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(z) + np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) * np.sin(2*np.pi*x) + np.sin(2 * 2*np.pi*z) * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) + 0.15
    elif label == 'Strut Tubular P':
        func = lambda x,y,z: 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)) - 5.1 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) + np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*z) * np.cos(2*np.pi*x)) - 14.6
    elif label == 'Strut Tubular G':
        func = lambda x,y,z: 10 * (np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.cos(2*np.pi*z) * np.sin(2*np.pi*x)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) - 14
    elif label == 'Strut I2-Y':
        func = lambda x,y,z: -2 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.sin(2*np.pi*x) * np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2 * 2*np.pi*z)) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*z)

    elif label == 'Strut BCC':
        func = lambda x,y,z: -2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))+(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z))


    elif label =='Sheet P':
        func = lambda x,y,z:(np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z))
    elif label == 'Sheet D':
        func = lambda x,y,z:(np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2*np.pi*z))
    elif label == 'Sheet G':
        func = lambda x,y,z: (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + np.sin(2*np.pi*z) * np.cos(2*np.pi*x) + np.sin(2*np.pi*y) * np.cos(2*np.pi*z))
    elif label == 'Sheet I-WP':
        func = lambda x,y,z: (2*(np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+np.cos(2*np.pi*y)*np.cos(2*np.pi*z)+np.cos(2*np.pi*z)*np.cos(2*np.pi*x))-(np.cos(2*2*np.pi*x)+np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*z)))
    elif label == 'Sheet F-RD':
        func = lambda x,y,z: (4 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) - (np.cos(2*2*np.pi*x)*np.cos(2*2*np.pi*y)+np.cos(2*2*np.pi*y)*np.cos(2*2*np.pi*z)+np.cos(2*2*np.pi*z)*np.cos(2*2*np.pi*x)))
    elif label == 'Sheet L':
        func = lambda x,y,z: (0.5 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(z) + np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) * np.sin(2*np.pi*x) + np.sin(2 * 2*np.pi*z) * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) + 0.15 )
    elif label == 'Sheet Tubular P':
        func = lambda x,y,z: (10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) + np.cos(2*np.pi*z)) - 5.1 * (np.cos(2*np.pi*x) * np.cos(2*np.pi*y) + np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*z) * np.cos(2*np.pi*x)) - 14.6)
    elif label == 'Sheet Tubular G':
        func = lambda x,y,z: (10 * (np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.cos(2*np.pi*z) * np.sin(2*np.pi*x)) - 0.5 * (np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*z) * np.cos(2 * 2*np.pi*x)) - 14)
    elif label == 'Sheet I2-Y':
        func = lambda x,y,z: (-2 * (np.sin(2 * 2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z) + np.sin(2*np.pi*x) * np.sin(2 * 2*np.pi*y) * np.cos(2*np.pi*z) + np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(2 * 2*np.pi*z)) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*y) + np.cos(2 * 2*np.pi*y) * np.cos(2 * 2*np.pi*z) + np.cos(2 * 2*np.pi*x) * np.cos(2 * 2*np.pi*z))
    return func

def create_TPMS_STL(func, output_filename = 'TPMS.stl', resolution = 100j, close = False, flip = True):
    """生成TPMSSTL面片格式

    Args:
        func (function): TPMS函数
    """

    res = resolution
    a = 1
    x, y, z = np.mgrid[0:a:res, 0:a:res, 0:a:res]

    # compute implicit function
    fun_values = func(x, y, z)
    import pyvista as pv
    # create grid for contouring
    grid = pv.StructuredGrid(x, y, z)
    grid["vol3"] = fun_values.ravel('F')
    contours3 = grid.contour([0])  # isosurface for 0
    if flip:
        contours3.flip_normals()
    # contours3.clip_box([0,1,0,1,0,1])
    if close:
        contour_caps = isocaps(x,y,z, fun_values, 0)
        appended = contours3.append_polydata(contour_caps)
        appended.save('data_STL/' + output_filename)
    else :
        contours3.save('data_STL/' + output_filename)
    return 'data_STL/' + output_filename

def create_Strut_TPMS_STL(func, output_filename = 'TPMS.stl'):
    """生成TPMSSTL杆结构格式

    Args:
        func (function): TPMS函数
    """
    res = 100j
    a = 1
    x, y, z = np.mgrid[0:a:res, 0:a:res, 0:a:res]
    fun_values = func(x, y, z)
    import pyvista as pv
    grid = pv.StructuredGrid(x, y, z)
    grid["vol3"] = fun_values.ravel('F')
    contours3 = grid.contour([0])  # isosurface for 0
    contours3.flip_normals()
    cube = pv.Cube((0.5,0.5,0.5),0.99,0.99,0.99).triangulate().subdivide(4, 'linear')
    closed = contours3.boolean_intersection(cube)
    closed.save('data_STL/' + output_filename)
    return 'data_STL/' + output_filename

def Dis2STL(distance,close = False, output_filename = 'Distance.stl'):

    resx = distance.shape[2]*1j
    resy = distance.shape[1]*1j
    resz = distance.shape[0]*1j
    a = 1
    x, y, z = np.mgrid[0:a:resx, 0:a:resy, 0:a:resz]
    import pyvista as pv
    grid = pv.StructuredGrid(x, y, z)
    grid["vol3"] = distance.transpose((2,1,0)).ravel('F')
    contours3 = grid.contour([0])  # isosurface for 0
    contours3.flip_normals()
    if close:
        cube = pv.Cube((0.5,0.5,0.5),0.99,0.99,0.99).triangulate().subdivide(4, 'linear')
        closed = contours3.boolean_intersection(cube)
        closed.save('out/' + output_filename)
    else:
        contours3.save('out/' + output_filename)
    return 'out/' + output_filename


def compute_output(func, mid):
    x, _ = TPMS2Mesh(60, 1, func, mid)
    return np.average(x)


from joblib import Parallel, delayed
def get_TPMS_const(label, density, lower_bound = -1, upper_bound = 1, cpu_cores = 1):
    if isinstance(label, str):
        func = get_TPMS_func(label)
    else:
        func = label

    # ! [1] Xu W, Zhang P, Yu M, et al. Topology Optimization Via Spatially-Varying TPMS[J]. IEEE Transactions on Visualization and Computer Graphics, 2023: 1-18.
    if label == 'Strut G':  # * density \in [0.017,0.983]
        return (density-0.5)/0.3337
    elif label == 'Strut P':  # * density \in [0.216,0.784]
        return (density-0.5)/0.2863
    elif label == 'Strut D':  # * density \in [0.082,0.918]
        return (density-0.5)/0.5883
    elif label == 'Strut I-WP':  # * density \in [0.012,0.897]
        return -2.692*density**5-0.664*density**4+5.176**density**3+0.3575**density**2+4.47*density-3.107
    elif label == 'Sheet G':  # * density \in [0,0.964]
        return density/0.6805
    elif label == 'Sheet P':  # * density \in [0,0.568]
        return density/0.5765
    elif label == 'Sheet D':  # * density \in [0,0.836]
        return density/1.1904
    elif label == 'Sheet I-WP':  # * density \in [0,0.885]
        return -0.537*density**3 + 3.828*density

    tolerance = 0.005
    while abs(upper_bound - lower_bound) > tolerance:
        mids = lower_bound + np.arange(1, cpu_cores + 1) * (upper_bound - lower_bound) / (cpu_cores + 1)
        xss = Parallel(n_jobs=cpu_cores)(delayed(compute_output)(func, mid) for mid in mids)
        index = np.searchsorted(xss, density)
        if index == 0:
            upper_bound = mids[index]
        elif index == len(mids):
            lower_bound =mids[index - 1]
        else:
            lower_bound = mids[index - 1]
            upper_bound = mids[index]

    return (upper_bound + lower_bound)/2

def create_continuous_TPMS_mesh(func_matrix, mesh_size = 40, k = 30):
    nelz,nely,nelx = func_matrix.shape
    x_all = np.zeros((mesh_size*nelz,mesh_size*nely,mesh_size*nelx))
    for idz in range(nelz):
        for idy in range(nely):
            for idx in range(nelx):
                func = func_matrix[idz,idy,idx]
                if idy == 0:
                    func1 = func
                else:
                    func1 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - y)))) * func(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - y))) * (func_matrix[idz,max(idy-1,0),idx](x,y,z))
                if idy == nely-1:
                    func2 = func1
                else:
                    func2 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (y - 1)))) * func1(x,y,z) \
                                    + 1/(1 + np.exp(- k * (y - 1))) * (func_matrix[idz,min(idy+1,nely-1),idx](x,y,z))
                if idx == 0:
                    func3 = func2
                else:
                    func3 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - x)))) * func2(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - x))) * (func_matrix[idz,idy,max(idx-1,0)](x,y,z))
                if idx == nelx-1:
                    func4 = func3
                else:
                    func4 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (x - 1)))) * func3(x,y,z) \
                                    + 1/(1 + np.exp(- k * (x - 1))) * (func_matrix[idz,idy,min(idx+1,nelx-1)](x,y,z))
                if idz == 0:
                    func5 = func4
                else:
                    func5 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - z)))) * func4(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - z))) * (func_matrix[max(idz-1,0),idy,idx](x,y,z))
                if idz == nelz-1:
                    func6 = func5
                else:
                    func6 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (z - 1)))) * func5(x,y,z) \
                                    + 1/(1 + np.exp(- k * (z - 1))) * (func_matrix[min(idz+1,nelz-1),idy,idx](x,y,z))
                x1,_ = TPMS2Mesh(mesh_size,1,func6,0)
                x_all[mesh_size*idz:mesh_size*(idz+1),mesh_size*idy:mesh_size*(idy+1),mesh_size*idx:mesh_size*(idx+1)] = x1
    return x_all

def create_continuous_TPMS_STL(func_matrix, k = 30):
    nelz,nely,nelx = func_matrix.shape
    # x_all = np.zeros((mesh_size*nelz,mesh_size*nely,mesh_size*nelx))

    apd = vtk.vtkAppendPolyData()

    for idz in range(nelz):
        for idy in range(nely):
            for idx in range(nelx):
                func = func_matrix[idz,idy,idx]
                if idy == 0:
                    func1 = func
                else:
                    func1 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - y)))) * func(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - y))) * (func_matrix[idz,max(idy-1,0),idx](x,y,z))
                if idy == nely-1:
                    func2 = func1
                else:
                    func2 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (y - 1)))) * func1(x,y,z) \
                                    + 1/(1 + np.exp(- k * (y - 1))) * (func_matrix[idz,min(idy+1,nely-1),idx](x,y,z))
                if idx == 0:
                    func3 = func2
                else:
                    func3 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - x)))) * func2(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - x))) * (func_matrix[idz,idy,max(idx-1,0)](x,y,z))
                if idx == nelx-1:
                    func4 = func3
                else:
                    func4 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (x - 1)))) * func3(x,y,z) \
                                    + 1/(1 + np.exp(- k * (x - 1))) * (func_matrix[idz,idy,min(idx+1,nelx-1)](x,y,z))
                if idz == 0:
                    func5 = func4
                else:
                    func5 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (0 - z)))) * func4(x,y,z) \
                                    + 1/(1 + np.exp(- k * (0 - z))) * (func_matrix[max(idz-1,0),idy,idx](x,y,z))
                if idz == nelz-1:
                    func6 = func5
                else:
                    func6 = lambda x,y,z:(1 - 1/(1 + np.exp(- k * (z - 1)))) * func5(x,y,z) \
                                    + 1/(1 + np.exp(- k * (z - 1))) * (func_matrix[min(idz+1,nelz-1),idy,idx](x,y,z))
                STL_filename = create_TPMS_STL(func6,'TPMS_%d_%d_%d.stl'%(idx,idy,idz))
                reader = vtk.vtkSTLReader()
                reader.SetFileName(STL_filename)
                reader.Update()
                pTransform = vtk.vtkTransform()
                pTransform.Translate((idx,idy,idz))
                pTransformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
                pTransformPolyDataFilter.SetInputData(reader.GetOutput())
                pTransformPolyDataFilter.SetTransform(pTransform)
                pTransformPolyDataFilter.Update()
                apd.AddInputData(pTransformPolyDataFilter.GetOutput())
                # x_all[mesh_size*idz:mesh_size*(idz+1),mesh_size*idy:mesh_size*(idy+1),mesh_size*idx:mesh_size*(idx+1)] = x1

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(apd.GetOutputPort())
    clean_filter.Update()
    writer = vtk.vtkSTLWriter()
    output_filename = "out/combine.stl"
    writer.SetFileName(output_filename)
    writer.SetInputConnection(clean_filter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    return output_filename

def create_continuous_TPMS_STLV2(xdata, ydata, zdata,start_point =(0,0,0), k = 30):
    nelz,nely,nelx = xdata.shape
    # nelz,nely,nelx = 1,3,3
    #! 适用于两个结构直接拼合，TPMS 1 和TPMS 2 结构分别过渡

    func1  = get_TPMS_func('Sheet P')
    func2  = get_TPMS_func('Sheet I-WP')
    rouP = xdata * ydata
    rouW = xdata * (1 - ydata)
    t_p = get_TPMS_const('Sheet P',rouP)
    t_g = get_TPMS_const('Sheet I-WP',rouW)

    existEle = np.where(xdata>0)
    existEle_line = existEle[0]*nelx*nely+existEle[2]*nely+nely-1-existEle[1]
    existEle_line_list = list(existEle_line)

    res = 100j

    z, y, x = np.mgrid[start_point[2]:start_point[2]+1:res, start_point[1]:start_point[1]+1:res, start_point[0]:start_point[0]+1:res]
    z_o, y_o, x_o = z.copy(),y.copy(),x.copy()
    fun_values1 = np.zeros_like(z)
    fun_values2 = np.zeros_like(z)
    res = 100

    index = existEle_line_list.index(start_point[2]*nelx*nely+start_point[0]*nely+nely-1-start_point[1])
    for idz in range(max(0,start_point[2]-2),min(nelz,start_point[2]+3)):
        for idy in range(max(0,start_point[1]-2),min(nely,start_point[1]+3)):
            for idx in range(max(0,start_point[0]-2),min(nelx,start_point[0]+3)):
                if xdata[idz,idy,idx] == 0:
                    continue

                x_o,y_o,z_o = \
                np.cos(zdata[idz,idy,idx])*(x-idx-0.5)+np.sin(zdata[idz,idy,idx])*(y-idy-0.5)+idx+0.5, \
                            -np.sin(zdata[idz,idy,idx])*(x-idx-0.5)+np.cos(zdata[idz,idy,idx])*(y-idy-0.5)+idy+0.5,\
                            z
                fun_values1 += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * (func1(x_o,y_o,z_o) - t_p[idz,idy,idx])
                fun_values2 += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * (func2(x_o,y_o,z_o) - t_g[idz,idy,idx])
    import pyvista as pv
    filename1 = ''
    filename2 = ''
    # create grid for contouring
    if rouP[start_point[::-1]] > 0.1:
        grid = pv.StructuredGrid(x, y, z)
        grid["volP"] = fun_values1.ravel('F')
        contoursP = grid.contour([0])  # isosurface for 0
        # contoursP.flip_normals()
        filename1 = 'data_STL/lattice_P_%d.stl'%(index)
        contoursP.save(filename1)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename1)
        reader.Update()
        decimationFilter = vtk.vtkDecimatePro()
        decimationFilter.SetInputData(reader.GetOutput())
        decimationFilter.SetTargetReduction(0.8) #Set#Specify the desired reduction in the total number of polygons
        decimationFilter.SetBoundaryVertexDeletion(False) #Turn on/off the deletion of vertices on the boundary of a mesh
        decimationFilter.PreserveTopologyOn()
        decimationFilter.Update()
        writer = vtk.vtkSTLWriter()

        writer.SetFileName(filename1)
        writer.SetInputConnection(decimationFilter.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.Update()
        writer.Write()
    if rouW[start_point[::-1]] > 0.1:
        filename2 = 'data_STL/lattice_W_%d.stl'%(index)
        grid = pv.StructuredGrid(x, y, z)
        grid["volW"] = fun_values2.ravel('F')
        contoursG = grid.contour([0])  # isosurface for 0
        # contoursG.flip_normals()
        # con = contoursG.boolean_union(contoursP)
        contoursG.save(filename2)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename2)
        reader.Update()
        decimationFilter = vtk.vtkDecimatePro()
        decimationFilter.SetInputData(reader.GetOutput())
        decimationFilter.SetTargetReduction(0.9) #Set#Specify the desired reduction in the total number of polygons
        decimationFilter.SetBoundaryVertexDeletion(False) #Turn on/off the deletion of vertices on the boundary of a mesh
        decimationFilter.PreserveTopologyOn()
        decimationFilter.Update()
        writer = vtk.vtkSTLWriter()
        # output_filename = "out/combine.stl"
        writer.SetFileName(filename2)
        writer.SetInputConnection(decimationFilter.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.Update()
        writer.Write()

    return filename1, filename2


def get_cap_faces(xdata, idx, idy, idz):
    faces = [0, 0, 0, 0, 0, 0]
    nelz,nely,nelx = xdata.shape
    if idx == 0 or xdata[idz, idy, idx-1] == 0:
        faces[0] = 1
    if idx == nelx-1 or xdata[idz, idy, idx+1] == 0:
        faces[1] = 1
    if idy == 0 or xdata[idz, idy-1, idx] == 0:
        faces[2] = 1
    if idy == nely-1 or xdata[idz, idy+1, idx] == 0:
        faces[3] = 1
    if idz == 0 or xdata[idz-1, idy, idx] == 0:
        faces[4] = 1
    if idz == nelz-1 or xdata[idz+1, idy, idx] == 0:
        faces[5] = 1

    return faces


def create_continuous_TPMS_STLV3(xdata, ydata, zdata, coefficient, start_point =(0,0,0), k = 30, TPMS1 = 'Sheet P', TPMS2 = 'Sheet D', res = 100, close = False, simplify_rate = 0, close_all =True):
    #! 适用于两个结构融合
    def poly5(a,x,y):
        return (a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2 +
        a[6]*x**3 + a[7]*x**2*y + a[8]*x*y**2 + a[9]*y**3 +
        a[10]*x**4 + a[11]*x**3*y + a[12]*x**2*y**2 + a[13]*x*y**3 + a[14]*y**4 +
        a[15]*x**5 + a[16]*x**4*y + a[17]*x**3*y**2 + a[18]*x**2*y**3 + a[19]*x*y**4 + a[20]*y**5)

    nelz, nely, nelx = xdata.shape
    func1 = get_TPMS_funcV2(TPMS1)
    func2 = get_TPMS_funcV2(TPMS2)
    existEle = np.where(xdata > 0)
    existEle_line = existEle[0]*nelx*nely+existEle[2]*nely+nely-1-existEle[1]
    existEle_line_list = list(existEle_line)

    resj = res*1j

    x, y, z = np.mgrid[start_point[0]:start_point[0]+1:resj, start_point[1]:start_point[1]+1:resj, start_point[2]:start_point[2]+1:resj]
    z_o, y_o, x_o = z.copy(),y.copy(),x.copy()
    fun_values = np.zeros_like(z)
    const_values = np.zeros_like(z)
    index = existEle_line_list.index(start_point[2]*nelx*nely+start_point[0]*nely+nely-1-start_point[1])
    for idz in range(max(0,start_point[2]-2),min(nelz,start_point[2]+3)):
        for idy in range(max(0,start_point[1]-2),min(nely,start_point[1]+3)):
            for idx in range(max(0,start_point[0]-2),min(nelx,start_point[0]+3)):
                if xdata[idz,idy,idx] == 0:
                    continue
                # BUG TODO 加入旋转之后 会导致连续性很差
                x_o,y_o,z_o = \
                np.cos(zdata[idz,idy,idx])*(x-idx-0.5)+np.sin(zdata[idz,idy,idx])*(y-idy-0.5)+idx+0.5, \
                            -np.sin(zdata[idz,idy,idx])*(x-idx-0.5)+np.cos(zdata[idz,idy,idx])*(y-idy-0.5)+idy+0.5,\
                            z
                fun_values += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * (ydata[idz,idy,idx]*func1(x_o,y_o,z_o)+(1-ydata[idz,idy,idx])*func2(x_o,y_o,z_o))
                const_values += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * poly5(coefficient,xdata[idz,idy,idx],ydata[idz,idy,idx])
                iso_values = (fun_values**2-const_values**2)

    # import pandas as pd
    # dataframe = pd.DataFrame({'x':x.reshape(-1),'y':y.reshape(-1),'z':z.reshape(-1),'s':fun_values.reshape(-1)})
    # dataframe.to_csv("test.csv",index=False,sep=',',header=False)
    # mesh2inp(iso_values,(0,1,0,1,0,1),(res,res,res))

    filename1 = ''
    # create grid for contouring
    if xdata[start_point[::-1]] > 0.05:

        grid = pv.StructuredGrid(x, y, z)
        grid["volP"] = iso_values.ravel('F')
        filename1 = 'data_STL/lattice_%d.stl' % (index)
        contours = grid.contour([0])   # isosurface for 0
        contours.flip_normals()
        if not close:
            contours.save(filename1)
        else:
            if not close_all:
                faces = get_cap_faces(xdata, start_point[0], start_point[1], start_point[2])
                contour_caps = isocaps(x,y,z,iso_values, 0, faces)
            else:
                contour_caps = isocaps(x,y,z,iso_values, 0)
            appended = contours.append_polydata(contour_caps)
            appended.save(filename1)

        if simplify_rate != 0:   # * 重建面片
            reader = vtk.vtkSTLReader()
            reader.SetFileName(filename1)
            reader.Update()
            # TODO 面片重建，现在实在是太烂了
            decimationFilter = vtk.vtkDecimatePro()
            decimationFilter.SetInputData(reader.GetOutput())
            decimationFilter.SetTargetReduction(simplify_rate)  # simplify_rate越大，面片越简化
            decimationFilter.SetBoundaryVertexDeletion(False)   #Turn on/off the deletion of vertices on the boundary of a mesh
            decimationFilter.PreserveTopologyOn()
            decimationFilter.Update()
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(filename1)
            writer.SetInputConnection(decimationFilter.GetOutputPort())
            writer.SetFileTypeToBinary()
            writer.Update()
            writer.Write()

    return filename1

def combine_STLV2(r_x,r_y,r_z,xdata,label = "lattice_",close = False, simplify_rate = 0.,output_filename='data_STL/combine.stl'):
    existEle = np.where(xdata > 0)
    nelz,nely,nelx = xdata.shape
    existEle_line = existEle[0]*nelx*nely+existEle[2]*nely+nely-1-existEle[1]
    existEle_line_list = list(existEle_line)
    apd = vtk.vtkAppendPolyData()
    for idx in r_x:
        for idy in r_y:
            for idz in r_z:
                if xdata[idz,idy,idx] != 0:
                    index = existEle_line_list.index(idz*nelx*nely+idx*nely+nely-1-idy)
                    filenamei = 'data_STL/%s%d.stl'%(label, index)
                    if os.path.exists(filenamei):
                        reader = vtk.vtkSTLReader()
                        reader.SetFileName(filenamei)
                        reader.Update()
                        position = [0,0,0]
                        pTransform = vtk.vtkTransform()
                        pTransform.Translate(position)
                        pTransformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
                        pTransformPolyDataFilter.SetInputData(reader.GetOutput())
                        pTransformPolyDataFilter.SetTransform(pTransform)
                        pTransformPolyDataFilter.Update()
                        apd.AddInputData(pTransformPolyDataFilter.GetOutput())
                    else:
                        print(filenamei + ' is not found.')
    apd.Update()
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(apd.GetOutputPort())
    clean_filter.Update()

    if simplify_rate != 0:   # * 重建面片
        # TODO 面片重建，现在实在是太烂了
        decimationFilter = vtk.vtkDecimatePro()
        decimationFilter.SetInputConnection(clean_filter.GetOutputPort())
        decimationFilter.SetTargetReduction(simplify_rate)  # simplify_rate越大，面片越简化
        decimationFilter.SetBoundaryVertexDeletion(False)   #Turn on/off the deletion of vertices on the boundary of a mesh
        decimationFilter.PreserveTopologyOn()
        decimationFilter.Update()
        clean_filter = decimationFilter


    if not close:
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(output_filename)
        writer.SetInputConnection(clean_filter.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.Update()
        writer.Write()
    else:   # to close the model
        tol = 0.03
        plane1 = vtk.vtkPlane()
        plane1.SetOrigin(tol ,0, 0)
        plane1.SetNormal(1.0, 0, 0)
        plane2 = vtk.vtkPlane()
        plane2.SetOrigin(nelx-tol ,0, 0)
        plane2.SetNormal(-1.0, 0, 0)
        plane3 = vtk.vtkPlane()
        plane3.SetOrigin(0, tol, 0)
        plane3.SetNormal(0, 1.0, 0)
        plane4 = vtk.vtkPlane()
        plane4.SetOrigin(0, nely-tol, 0)
        plane4.SetNormal(0, -1.0, 0)
        plane5 = vtk.vtkPlane()
        plane5.SetOrigin(0, 0, tol)
        plane5.SetNormal(0, 0, 1.0)
        plane6 = vtk.vtkPlane()
        plane6.SetOrigin(0, 0, nelz-tol)
        plane6.SetNormal(0, 0, -1.0)
        planes = vtk.vtkPlaneCollection()
        planes.AddItem(plane1)
        planes.AddItem(plane2)
        planes.AddItem(plane3)
        planes.AddItem(plane4)
        planes.AddItem(plane5)
        planes.AddItem(plane6)
        # 创建 vtkContourFilter 对象
        clipper = vtk.vtkClipClosedSurface()
        clipper.SetInputData(clean_filter.GetOutput())
        clipper.SetClippingPlanes(planes)
        clipper.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(output_filename)
        writer.SetFileTypeToBinary()
        writer.SetInputConnection(clipper.GetOutputPort())
        writer.Update()
        writer.Write()
    return output_filename


def create_continuous_TPMS_mesh(xdata, ydata, zdata, coefficient, start_point =(0,0,0), k = 30, TPMS1 = 'Sheet P', TPMS2 = 'Sheet D', res = 100, close = False, simplify_rate = 0, close_all =True):
    #! 适用于两个结构融合
    def poly5(a,x,y):
        return (a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2 +
        a[6]*x**3 + a[7]*x**2*y + a[8]*x*y**2 + a[9]*y**3 +
        a[10]*x**4 + a[11]*x**3*y + a[12]*x**2*y**2 + a[13]*x*y**3 + a[14]*y**4 +
        a[15]*x**5 + a[16]*x**4*y + a[17]*x**3*y**2 + a[18]*x**2*y**3 + a[19]*x*y**4 + a[20]*y**5)

    nelz, nely, nelx = xdata.shape
    func1 = get_TPMS_funcV2(TPMS1)
    func2 = get_TPMS_funcV2(TPMS2)
    existEle = np.where(xdata > 0)
    existEle_line = existEle[0]*nelx*nely+existEle[2]*nely+nely-1-existEle[1]
    existEle_line_list = list(existEle_line)

    resj = res*1j

    x, y, z = np.mgrid[start_point[0]:start_point[0]+1:resj, start_point[1]:start_point[1]+1:resj, start_point[2]:start_point[2]+1:resj]
    z_o, y_o, x_o = z.copy(),y.copy(),x.copy()
    fun_values = np.zeros_like(z)
    const_values = np.zeros_like(z)
    index = existEle_line_list.index(start_point[2]*nelx*nely+start_point[0]*nely+nely-1-start_point[1])
    for idz in range(max(0,start_point[2]-2),min(nelz,start_point[2]+3)):
        for idy in range(max(0,start_point[1]-2),min(nely,start_point[1]+3)):
            for idx in range(max(0,start_point[0]-2),min(nelx,start_point[0]+3)):
                if xdata[idz,idy,idx] == 0:
                    continue
                # BUG TODO 加入旋转之后 会导致连续性很差
                x_o,y_o,z_o = \
                np.cos(zdata[idz,idy,idx])*(x-idx-0.5)+np.sin(zdata[idz,idy,idx])*(y-idy-0.5)+idx+0.5, \
                            -np.sin(zdata[idz,idy,idx])*(x-idx-0.5)+np.cos(zdata[idz,idy,idx])*(y-idy-0.5)+idy+0.5,\
                            z
                fun_values += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * (ydata[idz,idy,idx]*func1(x_o,y_o,z_o)+(1-ydata[idz,idy,idx])*func2(x_o,y_o,z_o))
                const_values += np.exp(-k*((x_o-idx-0.5)**2 + (y_o-idy-0.5)**2 + (z_o-idz-0.5)**2)) * poly5(coefficient,xdata[idz,idy,idx],ydata[idz,idy,idx])
                iso_values = (fun_values**2-const_values**2)

    # mesh2inp(iso_values,(0,1,0,1,0,1),(res,res,res))
    return iso_values


def isocaps(x, y, z, iso_values, value, faces = [1,1,1,1,1,1]):
    pad = np.max(iso_values)
    len = iso_values.shape[1]
    res = len * 1j
    combine = pv.PolyData()

    if faces[0] == 1:
        # * x_min
        x_slice_min = np.concatenate((pad * np.ones_like(iso_values[0, :, :][np.newaxis,:,:]), iso_values[0, :, :][np.newaxis,:,:]), axis=0)
        x1, y1, z1 = np.mgrid[np.min(x):np.min(x):2j, np.min(y):np.max(y):res, np.min(z):np.max(z):res]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = x_slice_min.ravel('F')
        contoursxmin = grid.contour([0])
        contoursxmin.flip_normals()
        combine = combine.append_polydata(contoursxmin)

    if faces[1] == 1:
        # * x_max
        x_slice_max = np.concatenate((iso_values[len - 1, :, :][np.newaxis,:,:], pad * np.ones_like(iso_values[len - 1, :, :][np.newaxis,:,:])), axis=0)
        x1, y1, z1 = np.mgrid[np.max(x):np.max(x):2j, np.min(y):np.max(y):res, np.min(z):np.max(z):res]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = x_slice_max.ravel('F')
        contoursxmax = grid.contour([0])
        contoursxmax.flip_normals()
        combine = combine.append_polydata(contoursxmax)

    if faces[2] == 1:
        # * y_min
        y_slice_min = np.concatenate((pad * np.ones_like(iso_values[:, 0, :][:, np.newaxis, :]), iso_values[:, 0, :][:, np.newaxis, :]), axis=1)
        x1, y1, z1 = np.mgrid[np.min(x):np.max(x):res, np.min(y):np.min(y):2j, np.min(z):np.max(z):res]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = y_slice_min.ravel('F')
        contoursymin = grid.contour([0])
        contoursymin.flip_normals()
        combine = combine.append_polydata(contoursymin)

    if faces[3] == 1:
        # * y_max
        y_slice_max = np.concatenate((iso_values[:, len - 1, :][:, np.newaxis, :], pad * np.ones_like(iso_values[:, len - 1, :][:, np.newaxis, :])), axis=0)
        x1, y1, z1 = np.mgrid[np.min(x):np.max(x):res, np.max(y):np.max(y):2j, np.min(z):np.max(z):res]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = y_slice_max.ravel('F')
        contoursymax = grid.contour([0])
        contoursymax.flip_normals()
        combine = combine.append_polydata(contoursymax)

    if faces[4] == 1:
        # * z_min
        z_slice_min = np.concatenate((pad * np.ones_like(iso_values[:, :, 0][:, :, np.newaxis]), iso_values[:, :, 0][:, :, np.newaxis]), axis=2)
        x1, y1, z1 = np.mgrid[np.min(x):np.max(x):res, np.min(y):np.max(y):res, np.min(z):np.min(z):2j]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = z_slice_min.ravel('F')
        contourszmin = grid.contour([0])
        contourszmin.flip_normals()
        combine = combine.append_polydata(contourszmin)

    if faces[5] == 1:
        # * z_max
        z_slice_max = np.concatenate((iso_values[:, : , len - 1][:, :, np.newaxis], pad * np.ones_like(iso_values[:, : , len - 1][:, :, np.newaxis])), axis=2)
        x1, y1, z1 = np.mgrid[np.min(x):np.max(x):res, np.min(y):np.max(y):res, np.max(z):np.max(z):2j]
        grid = pv.StructuredGrid(x1, y1, z1)
        grid["vol"] = z_slice_max.ravel('F')
        contourszmax = grid.contour([0])
        contourszmax.flip_normals()
        combine = combine.append_polydata(contourszmax)


    return combine


def update_domain(y):
    nelz,nely,nelx = y.shape
    zmin = 0
    ymin = 0
    xmin = 0

    while np.sum(y[zmin,:,:]) == 0:
        zmin = zmin + 1
    while np.sum(y[:,ymin,:]) == 0:
        ymin = ymin + 1
    while np.sum(y[:,:,xmin]) == 0:
        xmin = xmin + 1
    while np.sum(y[nelz-1,:,:]) == 0:
        nelz = nelz - 1
    while np.sum(y[:,nely-1,:]) == 0:
        nely = nely - 1
    while np.sum(y[:,:,nelx-1]) == 0:
        nelx = nelx - 1

    return y[zmin:nelz,ymin:nely,xmin:nelx]


def combine_STL(data_filename = r'top_data/MechanicalClock.npy'):
    # data_filename = r'top_data/MechanicalClock.npy'
    data = np.load(data_filename,allow_pickle=True).item()
    x = data['variable'][0]
    y = data['variable'][1]
    nelx = x.shape[0]
    nely = x.shape[1]
    nelz = x.shape[2]

    existEle = np.where(x>0)
    existEle_line = existEle[0]*nelx*nely+existEle[2]*nely+nely-1-existEle[1]
    existEle_line_list = list(existEle_line)

    apd = vtk.vtkAppendPolyData()
    for idx in range(x.shape[2]):
        for idy in range(x.shape[1]):
            for idz in range(x.shape[0]):
                if x[idz,idy,idx] != 0:
                    index = existEle_line_list.index(idz*nelx*nely+idx*nely+nely-1-idy)
                    position = [idx, idy, idz]
                    # position = [0, 0, 0]
                    reader = vtk.vtkSTLReader()
                    filename1 = 'data_STL/lattice_%d.stl'%(index)
                    if os.path.exists(filename1):
                        reader.SetFileName(filename1)
                        reader.Update()
                        pTransform = vtk.vtkTransform()
                        pTransform.Translate(position)
                        pTransformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
                        pTransformPolyDataFilter.SetInputData(reader.GetOutput())
                        pTransformPolyDataFilter.SetTransform(pTransform)
                        pTransformPolyDataFilter.Update()
                        apd.AddInputData(pTransformPolyDataFilter.GetOutput())
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(apd.GetOutputPort())
    clean_filter.Update()
    writer = vtk.vtkSTLWriter()
    output_filename = "data_STL/combine.stl"
    writer.SetFileName(output_filename)
    writer.SetInputConnection(clean_filter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    return output_filename


def get_mesh_surface(voxels):
    """获得体素表面一层

    Args:
        voxels (n,n,n): Voxel Matrix
    """
    kernel = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])

    conv_result = ndimage.convolve(1 - voxels, kernel, mode='constant', cval=0.0)
    surface = np.zeros_like(voxels)
    surface[np.where((conv_result >= 1) & (voxels == 1))] = 1
    return surface


def get_mesh_surface2d(voxels, plain = 'xy'):
    """获得体素表面一层

    Args:
        voxels (n,n,n): Voxel Matrix
    """
    kernel = np.array([
        [0, 1, 0], [1, 0, 1], [0, 1, 0]
    ])
    surface = np.zeros_like(voxels)
    for idx in range(voxels.shape[0]):
        conv_result = ndimage.convolve(1 - voxels[idx], kernel, mode='constant', cval=0.0)
        surface[idx][np.where((conv_result >= 1) & (voxels[idx] > 0))] = 1
    return surface


def get_C0(E0 = 1, nu = 0.3):
    C0 = E0*1.0/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],[nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],[0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]])
    return C0


VTK_TO_ABAQUS_TYPE = {
    8: "S4R",
    9: "S4R",
    10: "C3D4",
    11: "C3D8R",
    12: "C3D8R"
}


def write_inp(FILENAME, nodes, elements, type='abaqus', BC_type='YF',force_size=1000):
    job_name = 'Job_1'
    model_name = 'Model_1'
    ele_offset = 1
    with open(FILENAME, "wt") as f:
        # ! abaqus
        if type == 'abaqus':
            f.write("*HEADING\n")
            f.write("**Job name: %s Model name: %s\n"%(job_name, model_name))
            f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
            f.write("*Part, name = PART - 1\n")
            # define node and element
            f.write('\n*Node, NSET=NALL\n')
            fmt = ", ".join(["{}"] + ["{:.9f}"] * nodes.shape[1]) + "\n"
            for k, x in enumerate(nodes):
                f.write(fmt.format(k + 1, *x))
            ele_nod = elements + ele_offset
            f.write('\n*Element, ELSET=EALL, type={}\n'.format(VTK_TO_ABAQUS_TYPE[12]))
            for e in range(1, elements.shape[0] + 1):
                f.write('{:d},  {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(e,*ele_nod[e-1,:]))

            # 定义全局集合 用于赋予材料特性
            f.write("*Elset, elset = Total, generate\n")
            f.write("1, {:d}, 1\n".format(elements.shape[0]))
            f.write("** Section: Section - 1\n")
            f.write("*Solid Section, elset = Total, material = Material - 1\n")
            f.write(",\n")
            f.write("*End Part\n")
            # Assembly
            f.write("*Assembly, name=Assembly\n")
            f.write("*Instance, name=PART-1-1, part=PART - 1\n")
            f.write("*End Instance\n")

            # ! force section set
            if BC_type == 'YF':
                f.write("*Nset, nset=Set-force, instance=PART-1-1\n")
                force_index = np.where(nodes[:, 1] == np.max(nodes[:, 1]))[0]
                force_nod_num = len(force_index)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index[force_nod_num // 12 * 12:force_nod_num] + 1)))
            elif BC_type == 'XX':
                f.write("*Nset, nset=Set-force1, instance=PART-1-1\n")
                force_index1 = np.where(nodes[:, 0] == np.min(nodes[:, 0]))[0]
                force_nod_num = len(force_index1)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index1[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index1[force_nod_num // 12 * 12:force_nod_num] + 1)))

                f.write("*Nset, nset=Set-force2, instance=PART-1-1\n")
                force_index2 = np.where(nodes[:, 0] == np.max(nodes[:, 0]))[0]
                force_nod_num = len(force_index2)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index2[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index2[force_nod_num // 12 * 12:force_nod_num] + 1)))

            elif BC_type == 'XXYY':
                f.write("*Nset, nset=Set-force1, instance=PART-1-1\n")
                force_index1 = np.where(nodes[:, 0] == np.min(nodes[:, 0]))[0]
                force_nod_num = len(force_index1)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index1[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index1[force_nod_num // 12 * 12:force_nod_num] + 1)))

                f.write("*Nset, nset=Set-force2, instance=PART-1-1\n")
                force_index2 = np.where(nodes[:, 0] == np.max(nodes[:, 0]))[0]
                force_nod_num = len(force_index2)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index2[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index2[force_nod_num // 12 * 12:force_nod_num] + 1)))

                f.write("*Nset, nset=Set-force3, instance=PART-1-1\n")
                force_index3 = np.where(nodes[:, 1] == np.min(nodes[:, 1]))[0]
                force_nod_num = len(force_index3)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index3[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index3[force_nod_num // 12 * 12:force_nod_num] + 1)))

                f.write("*Nset, nset=Set-force4, instance=PART-1-1\n")
                force_index4 = np.where(nodes[:, 1] == np.max(nodes[:, 1]))[0]
                force_nod_num = len(force_index4)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index4[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index4[force_nod_num // 12 * 12:force_nod_num] + 1)))

            elif BC_type == 'cantilever':
                f.write("*Nset, nset=Set-force, instance=PART-1-1\n")
                force_index = np.where(nodes[:, 0] == np.max(nodes[:, 0]))[0]
                force_nod_num = len(force_index)
                for i in range(force_nod_num // 12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(force_index[12 * i:12 * (i + 1)] + 1)))
                if force_nod_num - force_nod_num // 12 * 12 != 0:
                    str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(force_index[force_nod_num // 12 * 12:force_nod_num] + 1)))

            # ! fix section set
            if BC_type == 'YF':
                fix_index = np.where(nodes[:, 1] == np.min(nodes[:, 1]))[0]
            elif BC_type == 'XX':
                fix_index = np.where((nodes[:, 0] == (np.min(nodes[:, 0])+np.max(nodes[:, 0]))/2) & (((nodes[:, 1] == np.min(nodes[:, 1])) | (nodes[:, 1] == np.max(nodes[:, 1])))))[0]
            elif BC_type == 'cantilever':
                fix_index = np.where(nodes[:, 0] == np.min(nodes[:, 0]))[0]
            elif BC_type == 'XXYY':
                fix_index = np.where((np.abs(nodes[:, 0] - (np.min(nodes[:, 0])+np.max(nodes[:, 0]))/2) < 1) & (nodes[:, 1] == np.min(nodes[:, 1])))[0]

            if BC_type == 'YF' or BC_type == 'XXYY' or BC_type == 'XX' or BC_type == 'cantilever':
                fix_nod_num = len(fix_index)
                f.write("*Nset, nset=Set-fix, instance=PART-1-1\n")
                for i in range(fix_nod_num//12):
                    f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(*(fix_index[12*i:12*(i+1)]+1)))
                if fix_nod_num - fix_nod_num // 12 * 12 != 0:
                    str = "{:d},"*(fix_nod_num - fix_nod_num // 12 * 12) + "\n"
                    f.write(str.format(*(fix_index[fix_nod_num // 12 * 12:fix_nod_num]+1)))

            if 1:
                f.write("*Nset, nset=Set-fixZ, instance=PART-1-1, generate\n")
                f.write("1, {:d}, 1\n".format(nodes.shape[0]))

            f.write("*End Assembly\n")
            f.write("*Material, name=Material-1\n")
            f.write("*Elastic\n")
            f.write("2e+11, 0.3\n")
            f.write("*Step, name=Step-1, nlgeom=NO\n")
            f.write("*Static\n")
            f.write("1., 1., 1e-05, 1.\n")
            f.write("*Cload\n")
            # if BC_type == 'YF' or BC_type=='cantilever':
            #     f.write("Set-force, 2, 2, %.4f\n" % (-0.2 * np.max(nodes[:, 1])))
            # elif BC_type == 'XX':
            #     f.write("Set-force1, 1, 1, %.4f\n" % (0.1 * np.max(nodes[:, 1])))
            #     f.write("Set-force2, 1, 1, %.4f\n" % (-0.1 * np.max(nodes[:, 1])))
            # elif BC_type == 'XXYY':
            #     f.write("Set-force1, 1, 1, %.4f\n" % (0.1 * np.max(nodes[:, 1])))
            #     f.write("Set-force2, 1, 1, %.4f\n" % (-0.1 * np.max(nodes[:, 1])))
            #     f.write("Set-force3, 2, 2, %.4f\n" % (0.1 * np.max(nodes[:, 1])))
            #     f.write("Set-force4, 2, 2, %.4f\n" % (-0.1 * np.max(nodes[:, 1])))
            if BC_type == 'YF' or BC_type=='cantilever':
                f.write("Set-force, 2, %.8f\n" % (-force_size/len(force_index)))
            elif BC_type == 'XX':
                f.write("Set-force1, 1, %.8f\n" % (force_size/len(force_index1)))
                f.write("Set-force2, 1, %.8f\n" % (-force_size/len(force_index2)))
            elif BC_type == 'XXYY':
                f.write("Set-force1, 1, %.8f\n" % (force_size / len(force_index1)))
                f.write("Set-force2, 1, %.8f\n" % (-force_size / len(force_index2)))
                f.write("Set-force3, 2, %.8f\n" % (force_size / len(force_index3)))
                f.write("Set-force4, 2, %.8f\n" % (-force_size / len(force_index4)))


            if BC_type == 'YF' or BC_type == 'XX' or BC_type == 'XXYY' or BC_type=='cantilever':
                f.write("*Boundary\n")
                f.write("Set-fix, ENCASTRE\n") # 全固定
            if 1:
                f.write("*Boundary\n")
                f.write("Set-fixZ, 3, 3\n")
            f.write("** OUTPUT REQUESTS\n")
            f.write("*Restart, write, frequency=0\n")
            f.write("*Output, field, variable=PRESELECT\n")
            f.write("*Output, history, variable=PRESELECT\n")
            f.write("*End Step\n")

        # ! calculix
        elif type == 'calculix':
            f.write('\n*Node, NSET=NALL\n')
            fmt = ", ".join(["{}"] + ["{:.9f}"] * nodes.shape[1]) + "\n"
            for k, x in enumerate(nodes):
                f.write(fmt.format(k + 1, *x))
            ele_nod = elements + ele_offset
            f.write('\n*Element, ELSET=EALL, type={}\n'.format(VTK_TO_ABAQUS_TYPE[12]))
            for e in range(1, elements.shape[0] + 1):
                f.write('{:d},  {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(e, *ele_nod[e - 1, :]))
            # ~ fix section set
            f.write("*Nset, nset=FIX\n")
            fix_index = np.where(nodes[:, 1] == np.min(nodes[:, 1]))[0]
            fix_nod_num = len(fix_index)
            for i in range(fix_nod_num // 12):
                f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(
                    *(fix_index[12 * i:12 * (i + 1)] + 1)))
            if fix_nod_num - fix_nod_num // 12 * 12 != 0:
                str = "{:d}," * (fix_nod_num - fix_nod_num // 12 * 12) + "\n"
                f.write(str.format(*(fix_index[fix_nod_num // 12 * 12:fix_nod_num] + 1)))
            # ~ force section set
            f.write("*Nset, nset=FORCE\n")
            force_index = np.where(nodes[:, 1] == np.max(nodes[:, 1]))[0]
            force_nod_num = len(force_index)
            for i in range(force_nod_num // 12):
                f.write("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d},\n".format(
                    *(force_index[12 * i:12 * (i + 1)] + 1)))
            if force_nod_num - force_nod_num // 12 * 12 != 0:
                str = "{:d}," * (force_nod_num - force_nod_num // 12 * 12) + "\n"
                f.write(str.format(*(force_index[force_nod_num // 12 * 12:force_nod_num] + 1)))
            f.write("*Boundary\n")
            f.write("FIX, 1, 6\n")
            f.write("*MATERIAL,NAME=EL\n")
            f.write("*Elastic\n")
            f.write("2e+11, 0.3\n")
            f.write("*SOLID SECTION,ELSET=EALL,MATERIAL=EL\n")
            f.write("*Step\n")
            f.write("*Static,SOLVER=pardiso\n")
            # f.write("*Static\n")
            f.write("1., 1., 1e-05, 1.\n")
            # f.write("*Boundary\n")
            # f.write("FORCE, 2, 2, %.4f\n" % (-0.2 * np.max(nodes[:, 1])))
            f.write("*CLOAD\n")
            f.write("FORCE, 2, -10000\n")
            f.write("*NODE PRINT, NSET = NALL\n")
            f.write("U\n")
            f.write("*EL PRINT, ELSET = EALL\n")
            f.write("S\n")
            f.write("*el file\n")
            f.write("S\n")
            f.write("*node file\n")
            f.write("U\n")
            f.write("*END STEP\n")
    return


def write_vtk(FILENAME:str, type:int, nodes:np.ndarray, elements:np.ndarray):
    with open(FILENAME, "wt") as f:
        f.write(
            "# vtk DataFile Version 5.1\n"
            "Volume Mesh\n"
            "ASCII\n"
            "DATASET UNSTRUCTURED_GRID\n")

        f.write("POINTS  {:d}  float\n".format(nodes.shape[0]) )
        nodes.tofile(f, sep=" ")
        f.write("\n")

        f.write("CELLS {:d} {:d}\n".format(elements.shape[0]+1, elements.size))
        f.write("OFFSETS vtktypeint64\n")
        offsets = np.arange(0, elements.size+1, elements.shape[1], dtype=int)
        offsets.tofile(f, sep="\n")
        f.write("\n")

        f.write("CONNECTIVITY vtktypeint64\n")
        elements.tofile(f, sep="\n")
        f.write("\n")

        f.write("CELL_TYPES  {:d}\n".format(elements.shape[0]))
        np.full(elements.shape[0], type).tofile(f, sep="\n")
        f.write("\n")
    return


def write_vtu(FILENAME:str, type:int, nodes:np.ndarray, elements:np.ndarray):
    with open(FILENAME, "wt") as f:
        f.write(
            "<?xml version = \"1.0\"?>\n"
            "<VTKFile type = \"UnstructuredGrid\" version = \"0.1\" byte_order = \"LittleEndian\">\n"
            "<UnstructuredGrid>\n"
            "<Piece NumberOfPoints = \"{:d}\" NumberOfCells = \"{:d}\"> \n".format(nodes.shape[0], elements.shape[0])  )

        f.write(
            "<Points>\n"
            "<DataArray type = \"Float32\" Name = \"Points\" NumberOfComponents = \"3\" format = \"binary\">\n")
        data_bytes = nodes.astype(np.float32).tobytes()
        header = np.array(len(data_bytes), dtype=np.uint)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "</Points>\n"
            "<Cells>\n"
            "<DataArray type = \"Int32\" Name = \"connectivity\" format = \"binary\">\n")
        data_bytes = elements.astype(np.int32).tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "<DataArray type = \"Int32\" Name = \"offsets\" format = \"binary\">\n")
        offsets = np.arange(elements.shape[1], elements.size+1, elements.shape[1], dtype=np.int32)
        data_bytes = offsets.tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "<DataArray type = \"Int32\" Name = \"types\" format = \"binary\">\n")
        types = np.full(elements.shape[0], type).astype(np.int32)
        data_bytes = types.tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "</Cells>\n"
            "</Piece>\n"
            "</UnstructuredGrid>\n"
            "</VTKFile>\n")
    return


def mesh2inp(data, bounds=(0,1,0,1,0,1), mesh_size = (80,80,80), filename = 'out/voxel', type='abaqus',BC_type='YF',force_size=10000):
    data = - data
    NX,NY,NZ = mesh_size
    # compute the node vox matrix
    nod_vox = np.zeros((NX+1, NY+1, NZ+1))
    cube = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    for c in cube:
        nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] = nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] + data > 0

    _nnod = len(np.nonzero(nod_vox)[0])
    nod_vox = nod_vox > 0

    nod_order = np.zeros((NX+1, NY+1, NZ+1))
    # Generate the node matrix with global order
    nod_order[nod_vox] = range(_nnod)
    cube_g = np.array([[0,1,1],[0,0,1],[0,0,0],[0,1,0],[1,1,1],[1,0,1],[1,0,0],[1,1,0]])
    ele_nod_global = np.zeros((NX*NY*NZ,8), dtype='int32')
    # Loop all elements and record the nodes order
    for i in range(8):
        ele_nod_i = nod_order[cube_g[i][0]:NX+cube_g[i][0],:,:][:,cube_g[i][1]:NY+cube_g[i][1],:][:,:,cube_g[i][2]:NZ+cube_g[i][2]]
        ele_nod_global[:, i] = ele_nod_i.flatten()
    _ele_nod = ele_nod_global[(data > 0).flatten(), :]

    # compute the coordinates of nodes
    nod_x, nod_y, nod_z = np.meshgrid(range(NX+1), range(NY+1), range(NZ+1), indexing = 'ij')


    _nod_coor = np.zeros((_nnod, 3),dtype='int32')
    _nod_coor[:,0] = nod_x[nod_vox]
    _nod_coor[:,1] = nod_y[nod_vox]
    _nod_coor[:,2] = nod_z[nod_vox]

    dx = (bounds[1] - bounds[0]) / (NX)
    dy = (bounds[3] - bounds[2]) / (NY)
    dz = (bounds[5] - bounds[4]) / (NZ)

    lx = np.arange(bounds[0], bounds[1] + dx, dx)
    ly = np.arange(bounds[2], bounds[3] + dy, dy)
    lz = np.arange(bounds[4], bounds[5] + dz, dz)

    _nod_coor_abs = np.zeros((_nnod, 3),dtype=float)
    _nod_coor_abs[:, 0] = lx[_nod_coor[:, 0]]
    _nod_coor_abs[:, 1] = ly[_nod_coor[:, 1]]
    _nod_coor_abs[:, 2] = lz[_nod_coor[:, 2]]

    print('Number of voxels:',_ele_nod.shape[0])
    #write_inp/vtk/vtu/file
    write_inp(filename + "_" + type + '.inp', _nod_coor_abs, _ele_nod, type=type,BC_type=BC_type,force_size=force_size)
    # write_vtk(filename + "_" + type + '.vtk', 12, _nod_coor_abs, _ele_nod)


def get_x_dis_from_STL(stl_filename, cell_size, bounds):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filename)
    reader.Update()
    poly_data = reader.GetOutput()
    # bounds = poly_data.GetBounds()
    # bounds = (-20,20,-20,20,-1,1)
    cell_dims = np.ceil([(bounds[1] - bounds[0]) / cell_size,
                         (bounds[3] - bounds[2]) / cell_size,
                         (bounds[5] - bounds[4]) / cell_size]).astype(int)
    mesh_pitch = [cell_size, cell_size, cell_size]  # x, y, z
    mins = [bounds[0], bounds[2], bounds[4]]
    sdf1 = vtk.vtkImplicitPolyDataDistance()
    sdf1.SetInput(poly_data)

    xa = np.zeros((cell_dims[2], cell_dims[1], cell_dims[0]))
    dis = np.zeros((cell_dims[2], cell_dims[1], cell_dims[0]))
    for idx in range(cell_dims[0]):
        for idy in range(cell_dims[1]):
            for idz in range(cell_dims[2]):
                current_center = [(idx + 0.5) * cell_size + bounds[0], (idy + 0.5) * cell_size + bounds[2],
                                  (idz + 0.5) * cell_size + bounds[4]]
                sdf = sdf1.FunctionValue(current_center)
                xa[idz, idy, idx] = 1 if sdf < 0 else 0
                dis[idz, idy, idx] = sdf
    return xa, dis

def Rotate_STL(STL_filename, theta = 0, axis = 'z'):
    # 读取STL文件
    reader = vtk.vtkSTLReader()
    reader.SetFileName(STL_filename)
    reader.Update()

    # 创建一个旋转变换
    transform = vtk.vtkTransform()
    if axis == 'z':
        transform.RotateWXYZ(theta*180/np.pi, 0, 0, 1)
    elif axis == 'x':
        transform.RotateWXYZ(theta*180/np.pi, 1, 0, 0)
    elif axis == 'y':
        transform.RotateWXYZ(theta*180/np.pi, 0, 1, 0)

    # 创建一个变换滤波器
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    writer = vtk.vtkSTLWriter()
    output_filename = os.path.splitext(STL_filename)[0]+'_Rot%.2f.stl'%theta
    writer.SetFileName(output_filename)
    writer.SetInputConnection(transformFilter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    return output_filename

def process_iteration_STL(args):
    idx, idy, idz, xdata, ydata, zdata, res = args
    nelz, nely, nelx = xdata.shape
    # index = existEle_line_list.index(idz*nelx*nely+idx*nely+nely-1-idy)
    CPD_coefficient = [-0.009708, 0.8943, 0.4386, -1.041, 1.879, -4.147, 5.422, -11.59, -5.383, 10.23, -8.402, 8.535,
                       19.3, 5.841, -9.562, 3.104, 5.445, -14.53, -5.73, -1.67, 3.057]

    if xdata[idz, idy, idx] != 0:
        create_continuous_TPMS_STLV3(xdata, ydata, zdata, CPD_coefficient, (idx, idy, idz), k = 10, res=res, close = True, simplify_rate = 0, close_all = False)
        print(idz, idy, idx)


def process_iteration_Mesh(args):
    idx, idy, idz, xdata, ydata, zdata, res = args
    nelz, nely, nelx = xdata.shape
    CPD_coefficient = [-0.009708, 0.8943, 0.4386, -1.041, 1.879, -4.147, 5.422, -11.59, -5.383, 10.23, -8.402, 8.535,
                       19.3, 5.841, -9.562, 3.104, 5.445, -14.53, -5.73, -1.67, 3.057]
    if xdata[idz, idy, idx] != 0:
        iso_values = create_continuous_TPMS_mesh(xdata, ydata, zdata, CPD_coefficient, (idx, idy, idz), k=10, res=res, close=True, simplify_rate=0, close_all=False)
        print(idz, idy, idx)
        return iso_values


from multiprocessing import Pool
def Get_PD_continuous_STLorMesh(xdata, ydata, zdata, type='stl',cpu_cores=64, res=64,output_filename='data_STL/combine.stl',BC_type=None,force_size=10000):
    if type == 'stl':
        nelz, nely, nelx = xdata.shape
        existEle = np.where(xdata > 0)
        existEle_line = existEle[0] * nelx * nely + existEle[2] * nely + nely - 1 - existEle[1]
        existEle_line_list = list(existEle_line)

        display_range_x = range(0, nelx)
        display_range_y = range(0, nely)
        display_range_z = range(0, nelz)
        loop_parameters = [(idx, idy, idz, xdata, ydata, zdata, res) for idx in display_range_x for idy in
                       display_range_y for idz in display_range_z]

        # ! 并行计算
        with Pool(processes=cpu_cores) as pool:
            results = []
            for params in loop_parameters:
                result = pool.apply_async(process_iteration_STL, (params,))
                results.append(result)
            pool.close()
            pool.join()
        ofilename = combine_STLV2(display_range_x, display_range_y, display_range_z, xdata, close=False, simplify_rate=0.9, output_filename=output_filename)
        print(ofilename)

    elif type=='Mesh':
        nelz, nely, nelx = xdata.shape
        existEle = np.where(xdata > 0)
        existEle_line = existEle[0] * nelx * nely + existEle[2] * nely + nely - 1 - existEle[1]
        existEle_line_list = list(existEle_line)
        display_range_x = range(0, nelx)
        display_range_y = range(0, nely)
        display_range_z = range(0, nelz)
        loop_parameters = [(idx, idy, idz, xdata, ydata, zdata, res) for idx in display_range_x for idy in display_range_y for idz in display_range_z]

        Macro_mesh = np.zeros((nelx * res, nely * res, nelz * res))
        # ! 并行计算
        with Pool(processes=cpu_cores) as pool:
            results = []
            for params in loop_parameters:
                result = pool.apply_async(process_iteration_Mesh, (params,))
                results.append(result)
            pool.close()
            pool.join()
            for index, result in enumerate(results):
                params = loop_parameters[index]
                if xdata[params[2], params[1], params[0]] == 0:
                    continue
                Macro_mesh[res * params[0]:res * params[0] + res, res * params[1]:res * params[1] + res, res * params[2]:res * params[2] + res] = result.get()

        print(np.sum(Macro_mesh != 0))
        A = remove_largest_connected_region(Macro_mesh < 0)
        if np.max(A) != 0:
            print("Unconnected region")
            print(np.sum(A > 0) / np.sum(Macro_mesh < 0))
        else:
            print("Connected region")
        Macro_mesh[np.where(A > 0)] = 0
        B = remove_largest_connected_region(Macro_mesh < 0)
        print(np.sum(Macro_mesh < 0))
        mesh2inp(Macro_mesh, bounds=(0, nelx, 0, nely, 0, nelz), mesh_size=(res * nelx, res * nely, res * nelz), filename=output_filename, type='abaqus', BC_type=BC_type,force_size=force_size)
        print(output_filename)



if __name__ == '__main__':
    #x为数据矩阵，大小为(nelz,nely,nelx) 
    save_data(x,output_filename='out/volfrac')