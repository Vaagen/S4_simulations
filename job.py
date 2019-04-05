import numpy as np
import pandas as pd
import S4
from S4_wrapper import *
from S4_plotter import *
import argparse

def main(tot_num_processes=1, process_num=1):
    NumBasis = 300
    x_steps = 1
    y_steps = 1
    output_filename = 'output/test'
    df = setup_simulation(NumBasis, x_steps, y_steps)
    ###################################################
    plot_fields(df, 0, 100)
    ###################################################
    # pd.options.mode.chained_assignment = None  # default='warn'
    # if(tot_num_processes == 1):
    #     df = run_sim(df)
    #     # df = run_sim_intermediate_output(df,indexes=None,divisions=10, output_filename=output_filename)
    #     df.to_hdf(output_filename +'.h5', key='df', append=True)
    #     # print(df.loc[:,['tss_0','rss_0','phi_tss_0']])
    # else:
    #     run_sim_parallel(df,tot_num_processes=tot_num_processes, process_num=process_num, output_filename = output_filename +'_'+ str(process_num))
    # ###################################################
    # print('NB:  ', df.loc[0,'NumBasis'])
    # print('Dx:  ', df.loc[0,'Dx'])
    # print('Dy:  ', df.loc[0,'Dy'])
    # print('T:   ', df.loc[0,'tss_0']**2*np.sqrt(df.loc[0,'epsilon_fused_SiO2']))
    # print('Phi: ', (-df.loc[0,'phi_tss_0'] + np.pi)/(2*np.pi))
    ###################################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--processes", help="Total number of processes.")
    parser.add_argument("-n", "--process_number", help="Specific process number.")
    args = parser.parse_args()
    tot_num_processes = int(args.processes)
    process_num = int(args.process_number)
    print("Python job.py called with -p ", tot_num_processes, " -n ", process_num)
    main(tot_num_processes=tot_num_processes, process_num=process_num)
