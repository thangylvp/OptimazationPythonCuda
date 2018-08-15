from numba import jit, cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import time

num_par = 256
num_dim = 256
num_iter = 5000

TPB = num_dim
BPG = num_par
#global BPG # blocks per grid
#global TPB # threads per block

@cuda.jit
def _dev_calc_err(pos, err):
    
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    dev_shareMem = cuda.shared.array(shape = (TPB), dtype = float32)
    id = tx + ty * bw


    dev_shareMem[tx] = pos[ty, tx] * pos[ty, tx]
    cuda.syncthreads()
    if (tx == 0):
        err[ty] = 0.0
        for i in range(TPB):
            err[ty] += dev_shareMem[i]
        cuda.syncthreads()
    return

@cuda.jit
def _dev_update_par(rng_states, parPos, parBestPos, velPar, err, bestErr, gPos, bounds, W, C1, C2):
    
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    id = tx + ty * bw
    
    tmpBest = bestErr[ty]
    tmpErr = err[ty]

    if (tmpBest == -1 or tmpBest > tmpErr):
        tmpBest = tmpErr
        parBestPos[ty, tx] = parPos[ty, tx]
    
    if (tx == 0 ):
        bestErr[ty] = tmpBest

    r1 = xoroshiro128p_uniform_float32(rng_states, id)
    r2 = xoroshiro128p_uniform_float32(rng_states, id)

    #r1 = 0.3
    #r2 = 0.7

    vel_cognitive = C1 * r1 * (parBestPos[ty, tx] - parPos[ty, tx])
    vel_social = C2 * r2 * (gPos[tx] - parPos[ty, tx])

    velPar[ty, tx] = W * velPar[ty, tx] + vel_cognitive + vel_social

    parPos[ty, tx] = parPos[ty, tx] + velPar[ty, tx]

    if (parPos[ty, tx] < bounds[tx, 0]):
        parPos[ty, tx] = bounds[tx, 0]

    if (parPos[ty, tx] > bounds[tx, 1]):
        parPos[ty, tx] = bounds[tx, 1]

    

@jit(nopython=False)
def PSO(x0, bounds, num_particles, maxiter, W, C1, C2):

    num_dimensions = len(x0)
    TPB = num_dimensions
    BPG = num_particles    
    # print(TPB, BPG)
    err_best_g = -1.0  # best error for group
    pos_best_g = np.empty(shape = (num_dimensions), dtype = float)  # best position for group

    posPar = np.empty(shape = (num_particles, num_dimensions), dtype = float)
    posBestPar = np.empty(shape = (num_particles, num_dimensions), dtype = float)
    velPar = np.random.uniform(low = -1.0, high = 1.0, size = (num_particles, num_dimensions))
    posBestPar = np.empty(shape = (num_particles, num_dimensions), dtype = float)
    errBestPar = np.full((num_particles), -1.0)
    errPar = np.empty(shape = (num_particles), dtype = float)

    dev_posPar = cuda.device_array(shape = (num_particles, num_dimensions))
    dev_posBestPar = cuda.device_array(shape = (num_particles, num_dimensions))
    dev_errPar = cuda.device_array(shape = (num_particles))
    dev_velPar = cuda.device_array(shape = (num_particles, num_dimensions))
    dev_errBestPar = cuda.device_array(shape = (num_particles))
    dev_pos_best_g = cuda.device_array(shape = (num_dimensions))
    dev_bounds = cuda.device_array(shape = (num_dimensions, 2))
    
    dev_bounds = cuda.to_device(bounds)
    # g_pos_best = cuda.shared.array(shape = (TPB), dtype = float32)

    for i in range(num_particles):
        posPar[i] = x0
    print('--------------------')
    rng_states = cuda.random.create_xoroshiro128p_states(TPB * BPG, seed = 1)
    for Iter in range(maxiter):

        
        dev_posPar = cuda.to_device(posPar)
        _dev_calc_err[BPG, TPB](dev_posPar, dev_errPar)
        dev_errPar.copy_to_host(errPar)
        #print(errPar)
        #print(posPar)
        newPos = -1
        for j in range(num_particles):
            if (err_best_g == -1.0 or err_best_g > errPar[j]):
                err_best_g = errPar[j]
                newPos = j

        if (newPos != -1) :
            dev_posPar.copy_to_host(posPar)    
            pos_best_g = posPar[newPos]

        dev_pos_best_g = cuda.to_device(pos_best_g)
        dev_errBestPar = cuda.to_device(errBestPar)
        dev_velPar = cuda.to_device(velPar)
        
        

        _dev_update_par[BPG, TPB](rng_states, dev_posPar, dev_posBestPar, dev_velPar, dev_errPar, dev_errBestPar, dev_pos_best_g, dev_bounds, W, C1, C2)

        dev_posPar.copy_to_host(posPar)
        dev_posBestPar.copy_to_host(posBestPar)
        dev_velPar.copy_to_host(velPar)
        dev_errBestPar.copy_to_host(errBestPar)
        
        #print(posPar)
        # print final results
        
    print('--------------------')
    print(err_best_g)
    print(pos_best_g)
    return


# initial = [5, 5]  # initial starting location [x1,x2...]
# bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
# PSO(initial, bounds, num_particles=30, maxiter=10, W = 0.5, C1 = 1.0, C2 = 2.0)

#velPar = np.random.uniform(low = 2.0, high = 2.0, size = (10, 4))

#TPB = 4
#dev_velPar = cuda.to_device(velPar)
#dev_err = cuda.device_array(shape = (10))
#_dev_calc_err[10, 4](dev_velPar, dev_err)
#tmp = np.empty(shape = (10), dtype = float)
#dev_err.copy_to_host(tmp)

#print(velPar)
#print(tmp)
#print('------------')


initial = np.empty(shape = (num_dim), dtype = float)
bounds = np.empty(shape = (num_dim, 2))
for i in range(len(initial)):
    initial[i] = 5.0

for i in range(bounds.shape[0]):
    bounds[i, 0] = -10
    bounds[i, 1] = 10

print(initial)
print(bounds)
st = time.time()
PSO(initial, bounds, num_par, num_iter, W = 0.5, C1 = 1.0, C2 = 2.0)
en = time.time()

print(en - st)