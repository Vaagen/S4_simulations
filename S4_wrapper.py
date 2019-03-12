import numpy as np
import pandas as pd
import S4

from multiprocessing.dummy import Pool as ThreadPool
import itertools

import time

def run_sim(df, indexes=np.array([])):
    '''
    Run simulation for a set of different Dx, Dy, note that all other
    parameters are assumed to stay constant.
    Indexes indicates which simulations should be done, helps for parallelizing.
    '''
    if indexes.size == 0:
        indexes = range(df.index.size)
    ####################### Setup unit cell.
    a1 = [df.iloc[0].loc['a'],0]
    a2 = [df.iloc[0].loc['b']*np.cos(df.iloc[0].loc['angle_basis_vectors']), df.iloc[0].loc['b']*np.sin(df.iloc[0].loc['angle_basis_vectors'])]
    b1 = [0,0]
    b2 = [0,0]
    b1[0] = np.cos(df.iloc[0].loc['angle_shift_basis'])*a1[0] - np.sin(df.iloc[0].loc['angle_shift_basis'])*a1[1]
    b1[1] = np.cos(df.iloc[0].loc['angle_shift_basis'])*a1[1] + np.sin(df.iloc[0].loc['angle_shift_basis'])*a1[0]
    b2[0] = np.cos(df.iloc[0].loc['angle_shift_basis'])*a2[0] - np.sin(df.iloc[0].loc['angle_shift_basis'])*a2[1]
    b2[1] = np.cos(df.iloc[0].loc['angle_shift_basis'])*a2[1] + np.sin(df.iloc[0].loc['angle_shift_basis'])*a2[0]
    S = S4.New(Lattice=((b1[0],b1[1]),(b2[0],b2[1])), NumBasis=int(df.iloc[0].loc['NumBasis']))
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
    S.SetMaterial( Name='a_Si', Epsilon=df.iloc[0].loc['epsilon_a_Si'])
    S.SetMaterial( Name='fused_SiO2', Epsilon=df.iloc[0].loc['epsilon_fused_SiO2'])
    S.AddLayer(Name = 'AirAbove', Thickness = 0, Material = 'Air')
    S.AddLayer(Name = 'MetaLayer', Thickness = df.iloc[0].loc['z_Pillar'], Material = 'Air')
    S.AddLayer(Name = 'Substrate', Thickness = 0, Material = 'fused_SiO2')
    S.SetExcitationPlanewave(IncidenceAngles=(df.iloc[0].loc['theta_incidence'],df.iloc[0].loc['phi_incidence']),
        sAmplitude=df.iloc[0].loc['sAmplitude'], pAmplitude=df.iloc[0].loc['pAmplitude'], Order=0)
    S.SetFrequency(1/df.iloc[0].loc['wl'])
    Glist = S.GetBasisSet()
    num_modes = len(Glist)
    ####################### Calculation loop.
    for i in indexes:
        Dx,Dy = df.iloc[i].loc[['Dx','Dy']]
        ###################################################################
        S.RemoveLayerRegions(Layer='MetaLayer')
        # S.SetRegionEllipse(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.loc[i,'theta'], Halfwidths=(Dx/2.0,Dy/2.0))
        S.SetRegionRectangle(Layer='MetaLayer', Material='a_Si', Center=(0,0), Angle=df.iloc[i].loc['theta'], Halfwidths=(Dx/2.0,Dy/2.0))

        epsilon_fused_SiO2 = df.iloc[0].loc['epsilon_fused_SiO2']

        (forw_Amp_Substrate,back_Amp_Substrate) = S.GetAmplitudes(Layer = 'Substrate', zOffset = 0)
        (forw_Amp_Air,back_Amp_Air) = S.GetAmplitudes(Layer = 'AirAbove', zOffset = 0)

        if np.abs(df.iloc[0].loc['sAmplitude'])>1e-8:
            df.iloc[i, df.columns.get_loc('tss_0')] = np.abs(forw_Amp_Substrate[0]/np.sqrt(epsilon_fused_SiO2)/forw_Amp_Air[0])
            # df.loc[i,'tsp_0'] =np.abs(forw_Amp_Substrate[int(num_modes/2)]/np.sqrt(epsilon_fused_SiO2))/(np.abs(forw_Amp_Air[0]))
            df.iloc[i, df.columns.get_loc('rss_0')] = np.abs(back_Amp_Air[0]/forw_Amp_Air[0])
            # df.loc[i,'rsp_0'] = np.abs(back_Amp_Air[int(num_modes/2)])/np.abs(forw_Amp_Air[0])
            df.iloc[i, df.columns.get_loc('phi_tss_0')] = np.angle(forw_Amp_Air[0]/forw_Amp_Substrate[0]/np.sqrt(epsilon_fused_SiO2))
            # df.loc[i,'phi_tsp_0'] = np.angle(forw_Amp_Air[int(num_modes/2)]/forw_Amp_Substrate[0]/np.sqrt(epsilon_fused_SiO2))
        # if np.abs(df.loc[0,'pAmplitude'])>1e-8:
        #     df.loc[i,'tpp_0'] =
        #     df.loc[i,'tps_0'] =
        #     df.loc[i,'rpp_0'] =
        #     df.loc[i,'rps_0'] =
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


def run_sim_intermediate_output(df, indexes=None, divisions=10, output_filename='output/test.h5'):
    seconds_start = time.time()
    if indexes==None:
        indexes = np.arange(df.index.size)
    if(len(indexes)<divisions):
        divisions = len(indexes)
    indexes_array = np.array_split(indexes,divisions)
    for i in range(divisions):
        df = run_sim(df,indexes_array[i])
        df_out = df.iloc[indexes_array[i][0]:indexes_array[i][-1]+1]
        df_out.to_hdf(output_filename, key='df', append=True)
        seconds = time.time()-seconds_start
        hours   = np.floor(seconds/3600)
        minutes = np.floor((seconds - hours*3600)/60)
        seconds = seconds - hours*3600 - minutes*60
        print('{:.1f}% finished. \t{:.0f} h, {:.0f} min, {:.0f} sec.'.format((i+1)/divisions*100, hours, minutes, seconds))
    return df
