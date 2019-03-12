import numpy as np
import pandas as pd

from scipy.interpolate import interp2d
import scipy.interpolate
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import sys
import matplotlib.pyplot as plt

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
