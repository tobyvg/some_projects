import numba
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from matplotlib import animation

@numba.njit(parallel = True)
def cost_matrix_higher_dimensions(matrix1,matrix2,exponent = 2):
    shapes1 = np.array(matrix1.shape)
    shapes2 = np.array(matrix2.shape)
    prod1 = int(np.prod(shapes1))
    prod2 = int(np.prod(shapes2))
    vec1 = np.zeros((prod1,len(shapes1)))
    vec2 = np.zeros((prod2,len(shapes2)))
    size1 = vec1.shape[0]
    size2 = vec2.shape[0]
    for i in range(size1):
        index = np.zeros(len(shapes1))
        left = i
        for j in range(len(index)-1):
            times = (left//shapes1[-(j+1)])
            index[len(index)-2-j] = times
            subtract = left - shapes1[-(j+1)]*times
            index[len(index)-1-j] = subtract
            left = times
        vec1[i] = index.copy()
    for i in range(size2):
        index = np.zeros(len(shapes2))
        left = i
        for j in range(len(index)-1):
            times = (left//shapes2[-(j+1)])
            index[len(index)-2-j] = times
            subtract = left - shapes2[-(j+1)]*times
            index[len(index)-1-j] = subtract
            left = times
        vec2[i] = index.copy()
    M = np.zeros((size1,size2))
    for i in range(size1):
        for j in range(size2):
            M[i,j] = np.linalg.norm(vec1[i]-vec2[j],2)**exponent
    return M


def cost_matrix(vec_a,vec_b,exponent = 2,normalize = True,cost_matrix_higher_dimension = cost_matrix_higher_dimensions):
    if len(vec_a.shape) > 1 and len(vec_a.shape) > 1:
        M = cost_matrix_higher_dimensions(vec_a,vec_b,exponent = exponent)
        if normalize:
            M /= M.max()
        return M
    else:
        xa = np.arange(len(vec_a), dtype=np.float64)
        xb = np.arange(len(vec_b), dtype=np.float64)
        zeros = np.zeros((len(vec_a),len(vec_b)))
        for i in range(len(vec_a)):
            zeros[i,:] = np.abs(xb-xa[i])**exponent
        if normalize:
            zeros /= zeros.max()
        return zeros

def pre_calc_cost_matrix(dim1,dim2, exponent = 2,normalize = False, cost_matrix = cost_matrix):
    a = np.zeros(dim1)
    b = np.zeros(dim2)
    shape = list(a.shape)
    shape.extend(list(b.shape))
    M = cost_matrix(a,b,exponent = exponent,normalize = normalize).reshape(shape)
    return M
    del a
    del b

def cut_cost_matrix(matrix1,matrix2,M,old_shape = None, normalize = True):
    if (len(matrix1.shape) + len(matrix2.shape)) != len(M.shape):
        if old_shape == None:
            old_shape = np.ones(len(matrix1.shape)*2,dtype = int) * int(M.shape[0]**(1/len(matrix1.shape)))
        output = M.reshape(old_shape)
    else:
        output = M
    prod1 = np.prod(matrix1.shape)
    prod2 = np.prod(matrix2.shape)
    execute = 'output['
    for i in matrix1.shape:
        execute += ':' + str(i) + ','
    for i in range(len(matrix2.shape)):
        if i == len(matrix2.shape)-1:
            execute += ':' + str(matrix2.shape[i]) + ']'
        else:
            execute += ':' + str(matrix2.shape[i]) + ','
    output = eval(execute)
    output = output.reshape((prod1,prod2))
    if normalize:
        output /= output.max()
    return output

def normalize_function(x,params,function, offset = True,offset_zero= False,multiplier = 0.0001):
    array = function(x,params)
    if offset:
        if array.min() < 0:
            array -= array.min() ### To counter negative functions
    if offset_zero:
        array += array.max()*multiplier
    return array/np.sum(array)

def normalize_array(array,offset = True,offset_zero = False, multiplier = 0.0001):
    if offset:
        if array.min() < 0:
            array -= array.min() ### To counter negative functions
    if offset_zero:
        array += array.max()*multiplier
    return array/np.sum(array)

@numba.njit(parallel = True)
def sinkhorn_numba(a,b,M,lambd = 1e-3,maxiters = 10000, threshold = 10e-5,err_check = 10):
    f = a.copy()
    g = np.ones(len(b))
    K = np.exp((-M-lambd)/lambd)
    iters = maxiters
    err_log = 0
    errors = np.zeros(maxiters)
    sign = 1
    convergence = True
    ######
    #####
    #### Initialize Gamma
    gamma = np.zeros((len(a),len(b)))
    #gamma = np.dot(np.dot(np.diag(f),K),np.diag(g))
    for i in range(len(a)):
        for j in range(len(b)):
            gamma[i,j] = f[i]*g[j]*K[i,j]
    ####
    while sign*iters > 0: ### sign construct to avoid having two conditions in the while loop (unsupported by numba)
        g = b/(np.dot(K.T,f))
        f = a/(np.dot(K,g))
        #gamma = np.dot(np.dot(np.diag(f),K),np.diag(g))
        for i in range(len(a)):
            for j in range(len(b)):
                gamma[i,j] = f[i]*g[j]*K[i,j]
        ####################
        #####################
        diff = maxiters-iters
        if (diff/err_check-(diff)//err_check) == 0:
            error = np.linalg.norm(g*np.dot(K.T,f)-b,2)
            errors[err_log] = error
            sign = np.sign(error-threshold) ## Part of the sign construct
            err_log += 1
        iters -= 1
    if iters == 0:
        convergence = False
        print('Warning: max iterations exceeded, solution has not converged')
    errors = errors[:err_log]
    return gamma, np.sum(gamma*M),errors,convergence

def sinkhorn(a,b,M,lambd = 1e-3,maxiters = 10000, threshold = 10e-5,err_check = 10, sinkhorn_numba= sinkhorn_numba,verbose = True):
    start_time = time.time()
    if len(a.shape) > 1 or len(b.shape) > 1:
        vec_a = a.flatten()
        vec_b = b.flatten()
    else:
        vec_a = a
        vec_b = b
    if len(M.shape) != 2:
        print('For more efficiency please reshape the matrix to dimension 2 beforehand.')
        M_mat = M.reshape((vec_a.shape[0],vec_b.shape[0]))
    else:
        M_mat = M
    data = sinkhorn_numba(vec_a,vec_b,M_mat,lambd = lambd,maxiters = maxiters, threshold = threshold,err_check = err_check)
    if verbose and data[3]:
        print('Converged after '+ str(data[2].shape[0]*err_check) + ' iterations (' + str(time.time()-start_time) + ' seconds)' )
    return data

@numba.njit(parallel = True)
def transform(a,orig_shape,gamma,frame_number ,frames,indices,indices_provided =False):
    frac = frame_number/(frames)
    b_trans = a.copy()
    size = len(orig_shape)
    dummy = np.zeros(size+1,dtype = np.int64)
    multiplier = np.zeros(size + 1,dtype = np.int64)
    multiplier[0:-1] = orig_shape
    multiplier[-1] = 1
    pos_mem = np.zeros(size)
    if not indices_provided:
        indices = np.zeros((a.shape[0],len(orig_shape)))
        for i in range(gamma.shape[0]):
            left = i
            index = np.zeros(size)
            for k in range(index.shape[0]-1):
                times = (left//orig_shape[-(k+1)])
                index[index.shape[0]-2-k] = times
                subtract = left - orig_shape[-(k+1)]*times
                index[index.shape[0]-1-k] = subtract
                left = times
            indices[i] = index.copy()
    for i in range(gamma.shape[0]):
        pos_a = indices[i].copy()
        for j in range(gamma.shape[1]):
            if gamma[i,j] != 0:
                ### MASS TRANSPORT a->b
                b_trans[i] -= gamma[i,j]
                pos_b = indices[j].copy()
                average_pos = (frac*pos_b + (1-frac)*pos_a)
                ## Rounding
                pos = pos_mem.copy()
                for q in range(size):
                    pos[q] = int(np.around(average_pos[q]))
                ###
                cache = dummy.copy()
                cache[1:] = pos
                cache = cache*multiplier
                real_pos = np.sum(cache)
                b_trans[real_pos] += gamma[i,j]
    return  b_trans, indices
