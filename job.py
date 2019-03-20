import numpy as np
import pandas as pd
import S4
from S4_wrapper import *
import argparse

def main(tot_num_processes=1, process_num=1):
    pd.options.mode.chained_assignment = None  # default='warn'
    output_filename = 'output/test'
    df = setup_simulation()
    print(df.head())
    print(df.tail())
    if(tot_num_processes == 1):
        df = run_sim(df)
        # df = run_sim_intermediate_output(df,indexes=None,divisions=10, output_filename=output_filename)
        df.to_hdf(output_filename +'.h5', key='df', append=True)
        # print(df.loc[:,['tss_0','rss_0','phi_tss_0']])
    else:
        run_sim_parallel(df,tot_num_processes=tot_num_processes, process_num=process_num, output_filename = output_filename +'_'+ str(process_num))

def setup_simulation():
    # output_filename = 'output/arbabi_rectangle.h5'
    NumBasis =100# Comp time ~ NumBasis^3, Memory used ~ NumBasis^2
    # Use length micrometers
    # Wavelength
    wl = 0.915
    # Height of pillar
    z_Pillar = 0.715
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
    x_steps = 76
    y_steps = 76
    x_start = 0.065 #0.05
    y_start = 0.065 #0.05
    x_stop  = 0.44 #0.4
    y_stop  = 0.44 #0.4
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

    df = pd.DataFrame(data={'NumBasis':NumBasis, 'Dx':x_mesh, 'Dy':y_mesh, 'z_Pillar':z_Pillar,
        'theta':theta,'wl':wl,'a':a,'b':b,'angle_basis_vectors':angle_basis_vectors,'angle_shift_basis':angle_shift_basis,
        'epsilon_a_Si':epsilon_a_Si,'epsilon_fused_SiO2':epsilon_fused_SiO2,
        'sAmplitude':sAmplitude,'pAmplitude':pAmplitude,'theta_incidence':theta_incidence,'phi_incidence':phi_incidence,
        'tss_0':np.nan,
        'rss_0':np.nan,
        'phi_tss_0':np.nan})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--processes", help="Total number of processes.")
    parser.add_argument("-n", "--process_number", help="Specific process number.")
    args = parser.parse_args()
    tot_num_processes = int(args.processes)
    process_num = int(args.process_number)
    print("Python job.py called with -p ", tot_num_processes, " -n ", process_num)
    main(tot_num_processes=tot_num_processes, process_num=process_num)
