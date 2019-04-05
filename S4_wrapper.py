import numpy as np
import pandas as pd
import S4

from multiprocessing.dummy import Pool as ThreadPool
import itertools

import time

def setup_simulation(NumBasis, x_steps, y_steps):
    # output_filename = 'output/arbabi_rectangle.h5'
    NumBasis =NumBasis# Comp time ~ NumBasis^3, Memory used ~ NumBasis^2
    # Use length micrometers
    # Wavelength
    wl = 0.915
    # Height of pillar
    z_Pillar = 0.300
    # Pillar orientation
    theta = 0.0 # Degrees
    # Epsilon
    # epsilon_a_Si = 3.56**2.
    # epsilon_fused_SiO2 = 1.45**2.
    epsilon_a_Si = 13.921350491788342+0.17024349284921841j
    epsilon_fused_SiO2 = 2.121673706817246
    # Incoming amplitudes
    sAmplitude = 1.0
    pAmplitude = 1e-15
    # Incoming angles
    theta_incidence = 1e-15 # Polar angle in degrees
    phi_incidence   = 1e-15 # Azimuthal angle in degrees
    ################################## Geometries
    x_steps = x_steps
    y_steps = y_steps
    x_start = 0.065 #0.05
    y_start = 0.065 #0.2 #0.05
    x_stop  = 0.440 #0.1 #0.44 #0.4
    y_stop  = 0.440 #0.2 #0.44 #0.4
    ################################## Unit cell
    # Size of cell
    a = 0.65 #0.48
    b = a # Length of second lattice vector
    # # Rectangular basis
    # angle_basis_vectors = np.pi/180 * 90
    # angle_shift_basis   = np.pi/180 * 0
    # # Hexagonal basis
    angle_basis_vectors = np.pi/180 * 120
    angle_shift_basis   = np.pi/180 * 0
    ####################################################################
    x_values = np.linspace(x_start,x_stop, x_steps)
    y_values = np.linspace(y_start,y_stop, y_steps)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    x_mesh = np.ndarray.flatten(x_mesh)
    y_mesh = np.ndarray.flatten(y_mesh)

    # For using the mesh of x and y
    # Dx_values = x_mesh
    # Dy_values = y_mesh
    # For using each x/y value once
    Dx_values = x_values
    Dy_values = y_values

    df = pd.DataFrame(data={'NumBasis':NumBasis, 'Dx':Dx_values, 'Dy':Dy_values, 'z_Pillar':z_Pillar,
        'theta':theta,'wl':wl,'a':a,'b':b,'angle_basis_vectors':angle_basis_vectors,'angle_shift_basis':angle_shift_basis,
        'epsilon_a_Si':epsilon_a_Si,'epsilon_fused_SiO2':epsilon_fused_SiO2,
        'sAmplitude':sAmplitude,'pAmplitude':pAmplitude,'theta_incidence':theta_incidence,'phi_incidence':phi_incidence,
        'tss_0':np.nan,
        'rss_0':np.nan,
        'phi_tss_0':np.nan})
    return df

def setup_s4_instance(df, index):
    i = index
    ###################### Setup unit cell.
    a1 = [df.iloc[i, df.columns.get_loc('a')],0]
    a2 = [df.iloc[i, df.columns.get_loc('b')]*np.cos(df.iloc[i, df.columns.get_loc('angle_basis_vectors')]), df.iloc[i, df.columns.get_loc('b')]*np.sin(df.iloc[i, df.columns.get_loc('angle_basis_vectors')])]
    b1 = [0,0]
    b2 = [0,0]
    b1[0] = np.cos(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a1[0] - np.sin(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a1[1]
    b1[1] = np.cos(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a1[1] + np.sin(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a1[0]
    b2[0] = np.cos(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a2[0] - np.sin(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a2[1]
    b2[1] = np.cos(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a2[1] + np.sin(df.iloc[i, df.columns.get_loc('angle_shift_basis')])*a2[0]
    S = S4.New(Lattice=((b1[0],b1[1]),(b2[0],b2[1])), NumBasis=int(df.iloc[i, df.columns.get_loc('NumBasis')]))
    ####################### Options
    S.SetOptions(
        Verbosity = 0,
        LatticeTruncation = 'Circular', # Circular
        DiscretizedEpsilon = False, # False
        DiscretizationResolution = 8, # 8
        PolarizationDecomposition = True, # False
        PolarizationBasis = 'Jones', # Default
        LanczosSmoothing = False, # False
        SubpixelSmoothing = False, # False
        ConserveMemory = False # False
    )
    ####################### Setup material and exitation.
    S.SetMaterial( Name='Air', Epsilon=1)
    S.SetMaterial( Name='a_Si', Epsilon=df.iloc[i, df.columns.get_loc('epsilon_a_Si')])
    S.SetMaterial( Name='fused_SiO2', Epsilon=df.iloc[i, df.columns.get_loc('epsilon_fused_SiO2')])
    # Front illuminated: Illuminated from air
    # S.AddLayer(Name = 'AirAbove', Thickness = 0, Material = 'Air')
    # S.AddLayer(Name = 'MetaLayer', Thickness = df.iloc[i, df.columns.get_loc('z_Pillar')], Material = 'Air')
    # S.AddLayer(Name = 'Substrate', Thickness = 0, Material = 'fused_SiO2')
    # Back illuminated: Illuminated from substrate
    S.AddLayer(Name = 'Substrate', Thickness = 0, Material = 'fused_SiO2')
    S.AddLayer(Name = 'MetaLayer', Thickness = df.iloc[i, df.columns.get_loc('z_Pillar')], Material = 'Air')
    S.AddLayer(Name = 'AirAbove', Thickness = 0, Material = 'Air')
    #
    S.SetExcitationPlanewave(IncidenceAngles=(df.iloc[i, df.columns.get_loc('theta_incidence')],df.iloc[i, df.columns.get_loc('phi_incidence')]),
        sAmplitude=df.iloc[i, df.columns.get_loc('sAmplitude')], pAmplitude=df.iloc[i, df.columns.get_loc('pAmplitude')], Order=0)
    S.SetFrequency(1/df.iloc[i, df.columns.get_loc('wl')])
    return S

def run_sim(df, indexes=np.array([])):
    '''
    Run simulation for a set of different parameters.
    If scanning over Dx, Dy, run_sim is more efficient.
    Indexes indicates which simulations should be done, helps for parallelizing.
    '''
    if indexes.size == 0:
        indexes = range(df.index.size)
    ####################### Calculation loop.
    for i in indexes:
        S = setup_s4_instance(df,i)
        Glist = S.GetBasisSet()
        num_modes = len(Glist)
        # Do calculation
        Dx = df.iloc[i, df.columns.get_loc('Dx')]
        Dy = df.iloc[i, df.columns.get_loc('Dy')]
        ###################################################################
        S.RemoveLayerRegions(Layer='MetaLayer')
        S.SetRegionEllipse(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.iloc[i,df.columns.get_loc('theta')], Halfwidths=(Dx/2.0,Dy/2.0))
        # S.SetRegionRectangle(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.iloc[i,df.columns.get_loc('theta')], Halfwidths=(Dx/2.0,Dy/2.0))

        epsilon_fused_SiO2 = df.iloc[i, df.columns.get_loc('epsilon_fused_SiO2')]

        (forw_Amp_Substrate,back_Amp_Substrate) = S.GetAmplitudes(Layer = 'Substrate', zOffset = 0)
        (forw_Amp_Air,back_Amp_Air) = S.GetAmplitudes(Layer = 'AirAbove', zOffset = 0)

        # Front illuminated
        # if np.abs(df.iloc[i,df.columns.get_loc('sAmplitude')])>1e-8:
        #     df.iloc[i, df.columns.get_loc('tss_0')] = np.abs(forw_Amp_Substrate[0]/np.sqrt(epsilon_fused_SiO2)/forw_Amp_Air[0])
        #     df.iloc[i, df.columns.get_loc('rss_0')] = np.abs(back_Amp_Air[0]/forw_Amp_Air[0])
        #     df.iloc[i, df.columns.get_loc('phi_tss_0')] = np.angle(forw_Amp_Air[0]/forw_Amp_Substrate[0]*np.sqrt(epsilon_fused_SiO2))
        # Back illuminated
        if np.abs(df.iloc[i, df.columns.get_loc('sAmplitude')])>1e-8:
            df.iloc[i, df.columns.get_loc('tss_0')] = np.abs(forw_Amp_Air[0]/forw_Amp_Substrate[0])
            df.iloc[i, df.columns.get_loc('rss_0')] = np.abs(back_Amp_Substrate[0]/forw_Amp_Substrate[0])
            df.iloc[i, df.columns.get_loc('phi_tss_0')] = np.angle(forw_Amp_Substrate[0]/np.sqrt(epsilon_fused_SiO2)/forw_Amp_Air[0])
    S.GetFieldsOnGrid(z = 0.2, NumSamples=(4,4), Format = 'FileWrite', BaseFilename = 'field')
    return df

def run_sim_parallel(df,tot_num_processes=1, process_num=1, output_filename = 'output/test'):
    '''
    Run the simulation for a certain subset of the total domain.
    Save to a separate file so as to not mess with other calculations.
    '''
    indexes_array = np.array_split(np.arange(df.index.size),tot_num_processes)
    df_out = df.iloc[indexes_array[process_num], :]
    df_out = run_sim(df_out)
    df_out.to_hdf(output_filename +'.h5', key='df', append=True)

def get_fields(df, x,y,z, index=0):
    '''
    Gives fields at locations x, y and z,
    after solving with S4 for the case in df with index=index.
    x, y and z should be numpy lists of same length.
    '''
    a1 = [df.iloc[index, df.columns.get_loc('a')],0]
    a2 = [df.iloc[index, df.columns.get_loc('b')]*np.cos(df.iloc[index, df.columns.get_loc('angle_basis_vectors')]), df.iloc[index, df.columns.get_loc('b')]*np.sin(df.iloc[index, df.columns.get_loc('angle_basis_vectors')])]
    b1 = [0,0]
    b2 = [0,0]
    b1[0] = np.cos(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a1[0] - np.sin(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a1[1]
    b1[1] = np.cos(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a1[1] + np.sin(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a1[0]
    b2[0] = np.cos(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a2[0] - np.sin(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a2[1]
    b2[1] = np.cos(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a2[1] + np.sin(df.iloc[index, df.columns.get_loc('angle_shift_basis')])*a2[0]
    S = S4.New(Lattice=((b1[0],b1[1]),(b2[0],b2[1])), NumBasis=int(df.iloc[index, df.columns.get_loc('NumBasis')]))
    ####################### Options
    S.SetOptions(
        Verbosity = 0,
        LatticeTruncation = 'Circular', # Circular
        DiscretizedEpsilon = False, # False
        DiscretizationResolution = 8, # 8
        PolarizationDecomposition = True, # False
        PolarizationBasis = 'Jones', # Default
        LanczosSmoothing = False, # False
        SubpixelSmoothing = False, # False
        ConserveMemory = False # False
    )
    ####################### Setup material and exitation.
    S.SetMaterial( Name='Air', Epsilon=1)
    S.SetMaterial( Name='a_Si', Epsilon=df.iloc[index, df.columns.get_loc('epsilon_a_Si')])
    S.SetMaterial( Name='fused_SiO2', Epsilon=df.iloc[index, df.columns.get_loc('epsilon_fused_SiO2')])
    # Front illuminated: Illuminated from air
    # S.AddLayer(Name = 'AirAbove', Thickness = 0, Material = 'Air')
    # S.AddLayer(Name = 'MetaLayer', Thickness = df.iloc[0, df.columns.get_loc('z_Pillar')], Material = 'Air')
    # S.AddLayer(Name = 'Substrate', Thickness = 0, Material = 'fused_SiO2')
    # Back illuminated: Illuminated from substrate
    S.AddLayer(Name = 'Substrate', Thickness = 0, Material = 'fused_SiO2')
    S.AddLayer(Name = 'MetaLayer', Thickness = df.iloc[index, df.columns.get_loc('z_Pillar')], Material = 'Air')
    S.AddLayer(Name = 'AirAbove', Thickness = 0, Material = 'Air')
    #
    S.SetExcitationPlanewave(IncidenceAngles=(df.iloc[index, df.columns.get_loc('theta_incidence')],df.iloc[index, df.columns.get_loc('phi_incidence')]),
        sAmplitude=df.iloc[index, df.columns.get_loc('sAmplitude')], pAmplitude=df.iloc[index, df.columns.get_loc('pAmplitude')], Order=0)
    S.SetFrequency(1/df.iloc[index, df.columns.get_loc('wl')])
    Glist = S.GetBasisSet()
    num_modes = len(Glist)
    Dx = df.iloc[index, df.columns.get_loc('Dx')]
    Dy = df.iloc[index, df.columns.get_loc('Dy')]
    ###################################################################
    S.SetRegionEllipse(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.iloc[index,df.columns.get_loc('theta')], Halfwidths=(Dx/2.0,Dy/2.0))
    # S.SetRegionRectangle(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.iloc[index,df.columns.get_loc('theta')], Halfwidths=(Dx/2.0,Dy/2.0))
    ###################################################################
    E = np.zeros((len(x),3), dtype=np.complex)
    H = np.zeros((len(x),3), dtype=np.complex)
    for i in range(len(x)):
        (E[i],H[i]) = S.GetFields(x[i],y[i],z[i])
    return E,H
