import numpy as np
import pandas as pd

from scipy.interpolate import interp2d
import scipy.interpolate
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import sys
import matplotlib.pyplot as plt

from S4_wrapper import get_fields

def get_image_matrices(df, x_column, y_column, z_column):
    # Currently assumes square domain in x and y.
    data_m = df.loc[:,[x_column, y_column, z_column]].values
    data_x = np.flip(np.reshape(data_m[:,0], (int(np.sqrt(len(data_m[:,0]))),int(np.sqrt(len(data_m[:,0])))),order='C'),axis=0)
    data_y = np.flip(np.reshape(data_m[:,1], (int(np.sqrt(len(data_m[:,1]))),int(np.sqrt(len(data_m[:,1])))),order='C'),axis=0)
    data_z = np.flip(np.reshape(data_m[:,2], (int(np.sqrt(len(data_m[:,2]))),int(np.sqrt(len(data_m[:,2])))),order='C'),axis=0)
    return data_x, data_y, data_z

def plot_image(df, x_column, y_column, z_column, cmap='RdBu_r',limit=[0,1]):
    data_x,data_y,data_z = get_image_matrices(df, x_column, y_column, z_column)
    ## Using matplotlib
    fig, axs = plt.subplots(1, 1)
    ax = axs
    c = ax.pcolor(data_x, data_y, data_z, cmap=cmap,vmin=limit[0],vmax=limit[1])
    fig.colorbar(c, ax=ax)

def plot_image_interpolate(df, x_column, y_column, z_column, cmap='RdBu_r',limit=[0,1]):
    data_x,data_y,data_z = get_image_matrices(df, x_column, y_column, z_column)
    fig, axs = plt.subplots(1, 1)
    ax = axs
    # scipy interp. cubic
    f = interp2d(data_x[0,:], data_y[:,0], data_z,  kind='cubic')
    xnew = np.arange(np.min(data_x), np.max(data_x), 0.001)
    ynew = np.arange(np.min(data_y), np.max(data_y), 0.001)
    data_z1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    # Using matplotlib
    c = ax.pcolor(Xn,Yn, data_z1, cmap=cmap,vmin=limit[0],vmax=limit[1])
    fig.colorbar(c, ax=ax)

def plot_fields(df, index, resolution, axis='y', value=0):
    '''
    NOTE: Currently only working for axis='y' and value = 0
    Plot fields in cross section of unit cell for axis=value.
    Uses the index row of df.
    Plot becomes resolution^2 pixels large
    '''
    stepsize = 1/resolution
    if axis=='y':
        y = np.ones(resolution**2)*value
        x,z = np.mgrid[-0.5:0.5:stepsize, 0:1:stepsize]
        X,Z = np.mgrid[-0.5:0.5:stepsize, 0:1:stepsize]
        x = np.ndarray.flatten(x)*df.iloc[index, df.columns.get_loc('a')]
        z = np.ndarray.flatten(z)*df.iloc[index, df.columns.get_loc('z_Pillar')]*1.2
    else:
        return -1

    E,H = get_fields(df, x,y,z, index=0)
    Eabs = np.abs(E)
    Habs = np.abs(H)

    fig, axs = plt.subplots(2, 3)
    ax0 = axs[0,0]
    ax1 = axs[0,1]
    ax2 = axs[0,2]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    ax5 = axs[1,2]
    Y0 = np.reshape(Eabs[:,0], (resolution,resolution))
    Y1 = np.reshape(Eabs[:,1], (resolution,resolution))
    Y2 = np.reshape(Eabs[:,2], (resolution,resolution))
    Y3 = np.reshape(Habs[:,0], (resolution,resolution))
    Y4 = np.reshape(Habs[:,1], (resolution,resolution))
    Y5 = np.reshape(Habs[:,2], (resolution,resolution))
    c0 = ax0.pcolor(X, Z, Y0, cmap='RdBu_r',vmin=0,vmax=np.max(Eabs[:,0]))
    c1 = ax1.pcolor(X, Z, Y1, cmap='RdBu_r',vmin=0,vmax=np.max(Eabs[:,1]))
    c2 = ax2.pcolor(X, Z, Y2, cmap='RdBu_r',vmin=0,vmax=np.max(Eabs[:,2]))
    c3 = ax3.pcolor(X, Z, Y3, cmap='RdBu_r',vmin=0,vmax=np.max(Habs[:,0]))
    c4 = ax4.pcolor(X, Z, Y4, cmap='RdBu_r',vmin=0,vmax=np.max(Habs[:,1]))
    c5 = ax5.pcolor(X, Z, Y5, cmap='RdBu_r',vmin=0,vmax=np.max(Habs[:,2]))
    fig.colorbar(c0, ax=ax0)
    fig.colorbar(c1, ax=ax1)
    fig.colorbar(c2, ax=ax2)
    fig.colorbar(c3, ax=ax3)
    fig.colorbar(c4, ax=ax4)
    fig.colorbar(c5, ax=ax5)
    plt.show(block=False)
    input('Press enter to continue.')
