import cv2
from numba import cuda
import time
import math
import numpy as np
# GPU function
@cuda.jit()
def process_gpu(img):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for channel in range(3):
        color = img[tx, ty][channel] * 2.0 + 30
        if color > 255:
            img[tx, ty][channel] = 255
        elif color < 0:
            img[tx, ty][channel] = 0
        else:
            img[tx, ty][channel] = color


# CPU function
def process_cpu(img, dst):
    height, width, channels = img.shape
    for h in range(height):
        for w in range(width):
            for c in range(channels):
                color = img[h, w][c] * 2.0 + 30
                if color > 255:
                    dst[h, w][c] = 255
                elif color < 0:
                    dst[h, w][c] = 0
                else:
                    dst[h, w][c] = color


if __name__ == '__main__':
    img = np.zeros((100,100,3))
    height, width, channels = img.shape

    dst_cpu = img.copy()
    start_cpu = time.time()
    process_cpu(img, dst_cpu)
    end_cpu = time.time()
    time_cpu = (end_cpu - start_cpu)
    print("CPU process time: " + str(time_cpu))

    ##GPU function
    dImg = cuda.to_device(img)
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(height / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(width / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid, threadsperblock](dImg)
    end_gpu = time.time()
    cuda.synchronize()
    time_gpu = (end_gpu - start_gpu)
    print("GPU process time: " + str(time_gpu))
    dst_gpu = dImg.copy_to_host()

    # save
    cv2.imwrite("result_cpu.jpg", dst_cpu)
    cv2.imwrite("result_gpu.jpg", dst_gpu)
    print("Done.")