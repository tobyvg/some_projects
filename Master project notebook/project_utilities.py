import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import sys
from os import listdir
import re
import numba
import pickle
import time
from pyscf.tools import cubegen
from pyscf.dft import numint
from pyscf.tools.cubegen import Cube
from pyscf import lib

def spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def cartesian(spher):
    ptsnew = np.zeros(spher.shape)
    ptsnew[:,0] = spher[:,0]*np.sin(spher[:,1])*np.cos(spher[:,2])
    ptsnew[:,1] = spher[:,0]*np.sin(spher[:,1])*np.sin(spher[:,2])
    ptsnew[:,2] = spher[:,0]*np.cos(spher[:,1])
    return ptsnew

def flat_diag(M):
    shape = M.shape
    rows = []
    for i in range(shape[0]):
        rows.append(M[i,i:])
    return np.concatenate(rows)

def data2D(x,z):
    size = len(x)
    sq_size = int(np.sqrt(size))
    X = np.zeros(size,dtype = np.float64)
    Y = X.copy()
    Z = np.zeros(z.shape,dtype = np.float64)
    counter = int(0)
    for i,j in zip(x,z):
        X[counter] = i[0]
        Y[counter] = i[1]
        Z[counter] = j
        counter += int(1)
    X = X.reshape((sq_size,sq_size))
    Y = Y.reshape((sq_size,sq_size))
    Z = Z.reshape((sq_size,sq_size))
    return X,Y,Z


def argsort(array):
    a = array.copy()
    if len(a.shape) == 1:
        order = np.argsort(a)
        a = a[order]
    else:
        order = np.argsort(a[:,0])
        a = a[order]
        for i in range(1,a.shape[1]):
            mems = np.zeros((a.shape[0]-1,i))
            for p in range(i):
                mem = np.zeros(a.shape[0]-1)
                for k in range(1,a.shape[0]):
                    if a[k-1,p] == a[k,p]:
                        mem[k-1] = 1
                mems[:,p] = mem.copy()
            end = 0
            start_set = False
            for k in range(1,a.shape[0]):
                if 0 not in mems[k-1]:
                    end = k
                    if not start_set:
                        start = end -1
                        start_set = True
                    if k == a.shape[0]-1:
                        b = a[start: end+1]
                        order_cache = np.argsort(b[:,i])
                        a[start: end+1] = a[start:end+1][order_cache]
                        order[start: end+1] = order[start:end+1][order_cache]
                        start_set = False
                        end = 0
                elif end != 0 :
                    b = a[start: end+1]
                    order_cache = np.argsort(b[:,i])
                    a[start: end+1] = a[start:end+1][order_cache]
                    order[start: end+1] = order[start:end+1][order_cache]
                    start_set = False
                    end = 0
    return order



def splitat_(string, point,char = '_'):
    cache = ''
    counter = 0
    ret = ''
    found = False
    for i in string:
        if i != char:
            cache += i
        else:
            if counter == point:
                found = True
                ret = cache
            cache = ''
            counter += 1
    if not found:
        ret = cache
    return ret


def init_mpl(global_dpi,labelsize = 15.5,legendsize = 11.40, fontsize = 13,mat_settings = False): # 13
    if mat_settings:
        fontsize = 10
        labelsize = 13
    mpl.rcParams['figure.dpi']= global_dpi
    mpl.rc('axes', labelsize=labelsize)
    font = {'size'   : fontsize}#'family' : 'normal', #'weight' : 'bold'
    mpl.rc('font', **font)
    mpl.rc('legend',fontsize = legendsize)

atom_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def visualize_molecule(mol, atom_size = 1, bond_size = 1, plot = True, ax = None,plams = True):
    if plams:
        coords = [i.coords for i in mol]
        atnumbers = [i.atnum for i in mol]
        checklist = range(10)
        checklist = [str(i) for i in checklist]
        checklist.append('.')
        checklist.append('-')
        super_super_cache = []
        #Converting string of bond information to coordinates and bond order
        for i in mol.bonds:
            i = str(i)
            super_cache = []
            cache = []
            for k in i:
                if k in checklist:
                    cache.append(k)
                if k not in checklist and len(cache) > 0:
                    s = [str(j) for j in cache]
                    if s[0] == s[1] == '-':
                        s = s[2:-2]
            # Join list items using join()
                    res = float("".join(s))
                    super_cache.append(res)
                    cache = []
            super_super_cache.append(copy.deepcopy(super_cache))
    else:
        coords = mol.coords
        atnumbers = mol.atom_numbers()
    if ax == None:
        fig = plt.figure()
        ax = Axes3D(fig)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    color_dict = {}
    shape_dict = {}
    at_set = list(set(atnumbers))
    all_colors = ['C' + str(i) for i in range(10)]
    all_shapes = ['o','v','8','P','^','*','>','X','D','p','<','d']
    counter = 0
    at_set.sort()
    for i in at_set:
        len_cols = len(all_colors)
        len_shapes = len(all_shapes)
        corr_counter = counter - len_cols*(counter//len_cols)
        color_dict[i] = all_colors[corr_counter]
        shape_dict[i] = all_shapes[counter//len_cols-len_shapes*(counter//len_cols//len_shapes)]
        counter = counter + 1
    for i,k in zip(coords,atnumbers):
        size = atom_size*600*(1-np.exp(-k/7))
        ax.scatter(i[0], i[1], i[2], color = color_dict[k], s = size, marker = shape_dict[k])
    if plams:
        for i in super_super_cache:
            x = [i[0],i[4]]
            y = [i[1],i[5]]
            z = [i[2],i[6]]
            multi = i[3]
            ax.plot(x,y,z, color = 'black',linewidth = multi*2*bond_size, linestyle = (0,(1,0.5)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend_elements = []
    for i in at_set:
        if (i-1) < len(atom_symbols):
            legend_elements.append(Line2D([0], [0], marker=shape_dict[i], color='w', label= atom_symbols[i-1] + ' ('+str(i)+')',
                              markerfacecolor=color_dict[i], markersize=10))
        else:
            legend_elements.append(Line2D([0], [0], marker=shape_dict[i], color='w', label= '('+str(i)+')',
                              markerfacecolor=color_dict[i], markersize=10))
    if plams:
        bonds = [i[3] for i in super_super_cache]
        bonds = list(set(bonds))
        bonds.sort()
        for i in bonds:
            i = int(i)
            legend_elements.append(Line2D([0], [0], color='black', lw=i*2, label= 'Bond: ' + str(i),linestyle = (0,(1,0.5))))

    ax.legend(handles=legend_elements)

    if plot:
        plt.show()

######################## PLAMS ONLY ########################
def xyz_transformer(xyz): ## plams
    counter = 0
    super_cache = []
    cache = []
    for i in xyz:
        cache.append(i)
        counter += 1
        if counter >= 3:
            super_cache.append(tuple(cache))
            cache = []
            counter = 0
    return super_cache

def update_coords(mol,xyz,xyz_transformer = xyz_transformer, transform = True): #####plams
    mol1 = copy.deepcopy(mol)
    if transform:
        coords = xyz_transformer(xyz)
    else:
        coords = xyz
    for atom,xyz1 in zip(mol1,coords):
        atom.coords = xyz1
    return mol1


def swap_atoms(mol,indx1,indx2):
    at1 = mol.atoms[indx1]
    mol.atoms[indx1] = mol.atoms[indx2]
    mol.atoms[indx2] = at1
############################################## END PLAMS ONLY #########################




def visualize_matrix(matrix,color = mpl.cm.nipy_spectral, plot = True, vmin = None, vmax = None, ranges = None, rounding = 3, colorbar = True,fig_ax = None,x_points = 9, y_points =9,x_rotation = -45,labels = True,return_ax = False):
    if fig_ax == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    if ranges == None:
        img = ax.imshow(matrix, cmap = color, extent=[0,matrix.shape[1],matrix.shape[0],0], vmin = vmin, vmax = vmax)
    else:
        x_max = max(np.abs(ranges[0]),np.abs(ranges[1]))
        y_max = max(np.abs(ranges[2]),np.abs(ranges[3]))
        x_round = rounding - int(np.log10(x_max)//1)
        y_round = rounding - int(np.log10(y_max)//1)
        img = ax.imshow(matrix,cmap = color, vmin = vmin, vmax = vmax,extent=[-1,1,-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])])
        if len(matrix[0]) < x_points:
            ax.set_xticks(np.linspace(-1,1,x_points))
        else:
            ax.set_xticks(np.linspace(-1,1-2/len(matrix[0]),x_points))
        if len(matrix) < y_points:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0]),y_points))
        else:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])-
                                      len(matrix)/len(matrix[0])* 2/len(matrix),y_points))
        if x_round > 0 and x_round < 2*rounding:
            x = np.round(np.linspace(ranges[0],ranges[1],x_points),x_round)
        else:
            x = np.linspace(ranges[0],ranges[1],x_points)
            x = [np.format_float_scientific(i,precision = rounding-1) for i in x]
        if y_round > 0 and y_round < 2*rounding:
            y = np.round(np.linspace(ranges[3],ranges[2],y_points),y_round)
        else:
            y = np.linspace(ranges[3],ranges[2],y_points)
            y = [np.format_float_scientific(i,precision = rounding-1) for i in y]
        y = np.flip(y)
        ax.set_xticklabels(x,rotation = x_rotation)
        ax.set_yticklabels(y)
    if colorbar:
        fig.colorbar(img,ax = ax)
    if not labels:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    if plot:
        plt.show()
    if not plot and return_ax:
        return ax


class Mol:
    def __init__(self,atom_numbers = np.array([]),coords = np.array([]),name = None,sort = True):
        self.atom_symbols = atom_symbols
        #self.conversion = 1.8897259886 ## Conversion to Angstrom to Bohr
        #self.atnumbers = np.array([])
        self.coords = np.array(coords)
        self.atoms = np.array([self.atom_symbols[i-1] for i in atom_numbers])
        self.xyz_path = None
        self.name = name
        if sort:
            self.order_atoms()
    def read_xyz(self, xyz, conversion = False,sort = True):
        values = pd.read_csv(xyz,header = None,delim_whitespace=True, skiprows=2).values
        self.atoms = values[:,0]
        #self.atnumbers = np.array([self.atom_symbols.index(i) + 1 for i in self.atoms])
        coords = np.array(values[:,1:])
        if conversion:
            self.coords = coords * self.conversion
        else:
            self.coords = coords
        self.coords = self.coords.astype(np.float64)
        self.xyz_path = xyz
        if sort:
            self.order_atoms()
    def print_geometry(self):
        counter = 0
        for a,c in zip(self.atoms,self.coords):
            print('(' + str(counter) +'), '  +a + ': ' + str(c))
            counter += 1
    def atom_numbers(self):
        return np.array([self.atom_symbols.index(i) + 1 for i in self.atoms])
    def add_atom(self, atom_number, coord,sort = True):
        coords_old = self.coords.tolist()
        coords_old.append(coord)
        self.coords =np.array(coords_old)
        ######
        atoms_old = self.atoms.tolist()
        atoms_old.append(atom_symbols[atom_number-1])
        self.atoms = np.array(atoms_old)
        ######
        if sort:
            self.order_atoms()
    def add_atoms(self, atom_numbers, coords,sort = True):
        coords_old = self.coords.tolist()
        for i in coords:
            coords_old.append(i)
        self.coords =np.array(coords_old)
        ######
        atoms_old = self.atoms.tolist()
        for i in atom_numbers:
            atoms_old.append(atom_symbols[i-1])
        self.atoms = np.array(atoms_old)
        ######
        if sort:
            self.order_atoms()
    def update_geometry(self,atom_numbers,coords,sort = True):
        self.atoms = np.array([self.atom_symbols[i-1] for i in atom_numbers])
        self.coords = np.array(coords)
        if sort:
            self.order_atoms()
    def swap_atoms(self,indx1,indx2):
        cache1 = copy.deepcopy(self.coords[indx1])
        cache2 = copy.deepcopy(self.atoms[indx1])
        self.coords[indx1] = copy.deepcopy(self.coords[indx2])
        self.atoms[indx1] = copy.deepcopy(self.atoms[indx2])
        self.coords[indx2] = cache1
        self.atoms[indx2] = cache2
    def pyscf_out(self):
        output = []
        for a,c in zip(self.atoms, self.coords):
            output.append([a,tuple(c.tolist())])
        return output
    def psi4_out(self):
        output = """
        """
        last = False
        counter = 0
        s_list = []
        for a,c in zip(self.atoms, self.coords):
            output += """
            %s"""
            s = ''
            s += str(a)
            for i in c:
                s += ' ' + str(i)
            s_list.append(s)
        output = output % tuple(s_list)
        return output
    def open_ferm_out(self):
        return self.pyscf_out()
    def output_xyz(self, name = None,folder = None):
        foldername = folder
        if foldername == None:
            foldername  = 'Geometry_output'
        if name == None:
            if self.name == None:
                name = 'mol'
            else:
                name = self.name
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        counter = 0
        cache = foldername + '/' + name + '.xyz'
        while os.path.exists(cache):
            cache = foldername + '/' + name + '_' + str(counter+1) + '.xyz'
            counter = counter + 1
        towrite = ''
        towrite += str(len(self.coords)) + '\n'
        towrite += name
        for a,cs in zip(self.atoms,self.coords):
            towrite += '\n' + a
            for c in cs:
                towrite += '  ' + str(c)
        f = open(cache,'w')
        f.write(towrite)
        f.close()
        print('Geometry saved as: '+ cache)
    def order_atoms(self):
        atnums = self.atom_numbers()
        order = np.argsort(atnums)
        self.atoms = self.atoms[order]
        self.coords = self.coords[order]
    def vis_mol(self,atom_size = 1, bond_size = 1, plot = True, ax = None):
        coords = self.coords
        atnumbers = self.atom_numbers()
        if ax == None:
            fig = plt.figure()
            ax = Axes3D(fig)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        color_dict = {}
        shape_dict = {}
        at_set = list(set(atnumbers))
        all_colors = ['C' + str(i) for i in range(10)]
        all_shapes = ['o','v','8','P','^','*','>','X','D','p','<','d']
        counter = 0
        at_set.sort()
        for i in at_set:
            len_cols = len(all_colors)
            len_shapes = len(all_shapes)
            corr_counter = counter - len_cols*(counter//len_cols)
            color_dict[i] = all_colors[corr_counter]
            shape_dict[i] = all_shapes[counter//len_cols-len_shapes*(counter//len_cols//len_shapes)]
            counter = counter + 1
        for i,k in zip(coords,atnumbers):
            size = atom_size*600*(1-np.exp(-k/7))
            ax.scatter(i[0], i[1], i[2], color = color_dict[k], s = size, marker = shape_dict[k])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        legend_elements = []
        for i in at_set:
            if (i-1) < len(atom_symbols):
                legend_elements.append(Line2D([0], [0], marker=shape_dict[i], color='w', label= atom_symbols[i-1] + ' ('+str(i)+')',
                                  markerfacecolor=color_dict[i], markersize=10))
            else:
                legend_elements.append(Line2D([0], [0], marker=shape_dict[i], color='w', label= '('+str(i)+')',
                                  markerfacecolor=color_dict[i], markersize=10))
        ax.legend(handles=legend_elements)
        if plot:
            plt.show()
    def xyz_string(self):
        output = str(len(self.atoms))
        output += '\n\n'
        counter = 0
        for a,c in zip(self.atoms,self.coords):
            output += a
            for i in c:
                output += ' ' + str(i)
            if counter < len(self.atoms) - 1:
                output += '\n'
            counter += 1
        return output
    def rand_rotation(self):
        coords = self.coords
        X,Y,Z = self.rot_mats(random = True)
        for i in range(coords.shape[0]):
            coords[i] = X @ Y @ Z @ coords[i]
        self.coords = coords
    def rotation(self, x_rot = 0,y_rot = 0 ,z_rot =0 ):
        coords = self.coords
        X,Y,Z = self.rot_mats(x_rot,y_rot,z_rot)
        for i in range(coords.shape[0]):
            coords[i] = X @ Y @ Z @ coords[i]
        self.coords = coords
    def Zbond_center(self,index1,index2):
        origin = self.coords[index1]
        self.coords = self.coords-origin
        angle = self.coords[index2]
        angle = np.arctan(angle[1]/angle[2])
        self.rotation(x_rot = angle)
        angle = self.coords[index2]
        angle = np.arctan(angle[0]/angle[2])
        self.rotation(y_rot = -angle)
    def rotate_plane(self,index1,index2, angle = 0):
        self.Zbond_center(index1,index2)
        coords = self.coords
        z_rot = angle
        Z = np.array([[np.cos(z_rot), -np.sin(z_rot),0],[np.sin(z_rot), np.cos(z_rot),0], [0,0,1]])
        checker = (coords[index1] + coords[index2])/2
        for i in range(coords.shape[0]):
            if coords[i][-1] <= checker[-1]:
                coords[i] = Z @ coords[i]
        self.coords = coords
    def rot_mats(self, x_rot = 0,y_rot = 0 ,z_rot =0, random = False):
        if random:
            x_rot = np.random.uniform(-np.pi,np.pi)
            y_rot = np.random.uniform(-np.pi,np.pi)
            z_rot = np.random.uniform(-np.pi,np.pi)
        Z = np.array([[np.cos(z_rot), -np.sin(z_rot),0],[np.sin(z_rot), np.cos(z_rot),0], [0,0,1]])
        Y = np.array([[np.cos(y_rot),0,np.sin(y_rot)],[0,1,0],[-np.sin(y_rot),0,np.cos(y_rot)]])
        X = np.array([[1,0,0],[0,np.cos(x_rot),-np.sin(x_rot)],[0,np.sin(x_rot),np.cos(x_rot)]])
        return X, Y, Z



def eig(A):
    eigs,vecs = np.linalg.eig(A)
    order = np.argsort(eigs)
    eigs = eigs[order]
    vecs = vecs[:,order]
    return eigs,vecs


def save_obj(obj, name,folder = None ):
    if folder == None:
        if not os.path.exists('obj'):
            os.makedirs('obj')
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        if folder[-1] != '/':
            folder += '/'
        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ,folder = None):
    if folder == None:
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        if folder[-1] != '/':
            folder += '/'
        with open(folder+ name + '.pkl', 'rb') as f:
            return pickle.load(f)


def read_cube(path, verbose = False,convert_steps = True):
    if verbose:
        start_time = time.time()
    file = open(path,'r')
    data = file.readlines()[2:6]
    file.close()
    origin = data[0].replace('\n','')
    origin = np.fromstring(origin,dtype = float,count = 4, sep = ' ')
    num_atoms = origin[0]
    origin = origin[1:]
    x_step = data[1].replace('\n','')
    x_step = np.fromstring(x_step,dtype = float,count = 4, sep = ' ')
    x_steps = int(x_step[0])
    x_step = x_step[1:]
    y_step = data[2].replace('\n','')
    y_step = np.fromstring(y_step,dtype = float,count = 4, sep = ' ')
    y_steps = int(y_step[0])
    y_step = y_step[1:]
    z_step = data[3].replace('\n','')
    z_step = np.fromstring(z_step,dtype = float,count = 4, sep = ' ')
    z_steps = int(z_step[0])
    z_step = z_step[1:]
    start = 5+ 1 + int(num_atoms)
    file = open(path,'r')
    density_data = file.readlines()[start:]
    file.close()
    density = ' '.join(density_data)
    density = np.fromstring(density,sep = ' ')
    density = density.reshape((x_steps,y_steps,z_steps))
    if verbose:
        print('Execution time: reading density:')
        print(time.time()-start_time)
    steps = (origin,(x_steps,y_steps,z_steps),(x_step,y_step,z_step))
    if convert_steps:
        x_start,y_start,z_start = steps[0]
        x_end = (steps[1][0]-1)*steps[2][0][0] + x_start
        y_end = (steps[1][1]-1)*steps[2][1][1] + y_start
        z_end = (steps[1][2]-1)*steps[2][2][2] + z_start
        x = np.array([x_start,x_end,steps[1][0]])
        y = np.array([y_start,y_end,steps[1][1]])
        z = np.array([z_start,z_end,steps[1][2]])
        steps = np.array([y,x,z])
    return density,steps

def AO(mol,C,outfile = 'AO_dir/AOs', nx=80, ny=80, nz=80, resolution=None, save = True):
    counter = 1
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    else:
        exists = True
        while exists:
            cache = outfile + '_' + str(counter)
            exists = os.path.exists(cache)
            counter += 1
        outfile = cache
        os.makedirs(cache)
    cc = Cube(mol, nx, ny, nz, resolution)
    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    #blksize = min(8000, ngrids)
    blksize = ngrids
    rho = np.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
    x = (coords[:,0].min(),coords[:,0].max(),nx)
    y =(coords[:,1].min(),coords[:,1].max(),ny)
    z = (coords[:,2].min(),coords[:,2].max(),nz)
    if save:
        np.savetxt(outfile + '/AOs', ao)
        np.savetxt(outfile + '/C',C)

        np.savetxt(outfile+'/coords',np.array([y,x,z]))
        f = open(outfile + '/geom_basis','w')
        f.write(str(mol.atom) + '_')
        f.write(str(mol.basis))
        f.close()
        #cc.write(np.array([]),outfile + '_geom',comment = 'Geometry')
    return ao,np.array([y,x,z])


def load_AO(path):
    if path[-1] != '/':
        path = path +'/'
    ao = np.loadtxt(path + 'AOs')
    C = np.loadtxt(path + 'C')
    coords = np.loadtxt(path + 'coords')
    f = open(path + 'geom_basis','r')
    text = f.read()
    f.close()
    geom = eval(splitat_(text,0))
    basis = splitat_(text,1)
    mol = gto.M(atom=geom, basis=basis)
    return mol,C,coords,ao

@numba.njit(parallel=True)
def AOtoMO(ao,C, orbital,nx,ny,nz):
    MO = np.zeros((ao.shape[0]))
    for i in range(ao.shape[0]):
        MO[i] = np.dot(ao[i],  C[:,orbital])
    MO = MO.reshape((nx,ny,nz))
    return MO

def AOtorho(mol,ao,dm,nx,ny,nz):
    rho = np.zeros((ao.shape[0]))
    rho = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape((nx,ny,nz))
    return rho




def MO(mol,C,orbital,outfile = 'orbital', nx=80, ny=80, nz=80, resolution=None, save = True):
    """Calculates electron density and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """

    cc = Cube(mol, nx, ny, nz, resolution)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    #blksize = min(8000, ngrids)
    blksize = ngrids
    rho = np.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
    MO = np.array([i @ C[:,orbital] for i in ao])
    MO = MO.reshape((nx,ny,nz))
    if save:
        cc.write(MO, outfile, comment='Orbital '+ str(orbital) + ' in real space (e/Bohr^3)')
    x = (coords[:,0].min(),coords[:,0].max(),nx)
    y =(coords[:,1].min(),coords[:,1].max(),ny)
    z = (coords[:,2].min(),coords[:,2].max(),nz)
    return MO,np.array([y,x,z])

def density(mol, dm, outfile = 'rho', nx=80, ny=80, nz=80, resolution=None, save = True):
    """Calculates electron density and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    cc = Cube(mol, nx, ny, nz, resolution)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    #blksize = min(8000, ngrids)
    blksize = ngrids
    rho = np.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape((nx,ny,nz))
    if save:
        cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    x = (coords[:,0].min(),coords[:,0].max(),nx)
    y =(coords[:,1].min(),coords[:,1].max(),ny)
    z = (coords[:,2].min(),coords[:,2].max(),nz)
    return rho,np.array([y,x,z])

def dxdydz(coords):
    d = 1
    for i in coords:
        d *= (i[1]-i[0])/i[2]
    return d

def rdm1(C,e):
    return 2*C[:,:e] @ C.T[:e,:]

@numba.njit(parallel = True)
def set_zeros_thresh(M,threshold):
    A = M.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.abs(A[i,j]) < threshold:
                A[i,j] = 0
    return A


@numba.njit(parallel = True)
def normalize(M,SB):
    A = M.copy()
    for i in range(A.shape[0]): ### normalization
        A[:,i] /= np.sqrt(SB[i,i])
    return A

def train_test_split(x,y,test_size = 0.1 ,shuffle = False):
    size = x.shape[0]
    test_size = np.floor(test_size*size)
    train_size = int(size- test_size)
    if shuffle:
        shuffler= np.arange(size)
        np.random.shuffle(shuffler)
        return x[shuffler][:train_size],x[shuffler][train_size:],y[shuffler][:train_size],y[shuffler][train_size:]
    else:
        return x[:train_size],x[train_size:],y[:train_size],y[train_size:]

def water_straight(angle,length1,length2):
    angle = angle/360 *2*np.pi
    O_coord = np.array([0,0,0])
    H1_coord = np.array([length1,0,0])
    H2_coord = np.array([length2*np.cos(angle),length2*np.sin(angle),0])
    mol = Mol([8,1,1],[O_coord,H1_coord,H2_coord])
    return mol
