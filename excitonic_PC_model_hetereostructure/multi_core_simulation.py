import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
from parallelization_framework import get_absorption_coefficients, single_pulse_photoresponse, \
    two_pulses_photoresponse, photocurrent_delta_t_discrete_index, wavelength_to_energy
import sys
import pandas as pd
from datetime import datetime
import os


def read_args():
    """
    Reads arguments passed to the script in this order:
    NUMBER OF TASKS, ALPHA_0, ALPHA_1, GAMMA_0, GAMMA_1, TAU_0, TAU_1, ENERGY_0, ENERGY_1, POWER_0, POWER_1,
    TIME_RANGE, TIME_RESOLUTION, DELTA_T_SWEEP, DELTA_T_RESOLUTION, SAVE_INDEX
    """
    if len(sys.argv) > 1:
        tasks = int(sys.argv[1])
        alphas = np.array([sys.argv[2], sys.argv[3]])
        gammas = np.array([sys.argv[4], sys.argv[5]])
        taus = np.array([sys.argv[6], sys.argv[7]])
        energies = np.array([sys.argv[8], sys.argv[9]])
        powers = np.array([sys.argv[10], sys.argv[11]])
        time_range = (sys.argv[12], sys.argv[13])
        time_resolution = int(sys.argv[14])
        delta_t_range = (sys.argv[15], sys.argv[16])
        delta_t_res = sys.argv[17]
        save_index = sys.argv[18]

        return [tasks, alphas, gammas, taus, energies, powers, time_range, time_resolution, delta_t_range, delta_t_res,
                save_index]


def timestamp_maker():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


read_args()
# # NATURAL CONSTANTS
h = 6.62607015e-34
c = 2.99792458e8
e = 1.602176634e-19

########################################################################################################################
# https://www.desmos.com/calculator/mqnypi6npa
# MAIN SIMULATION TOGGLE
example_output = False
plot_extinction = False
pc_plot = False
save_data = True
savedir = f'.\\out'
save_index = 1
use_date_as_folder_name = False
folder_name = 'default'
number_of_tasks = 12
logscale = False
# DEVICE PARAMETERS - first entry is MoSe2 (top), second is MoS2 (bottom)
alpha = np.array([0e11, 0e11])  # coupling constants
tau = np.array([25, 225]) * 1e-12  # photoresponse time
gamma = np.array([0.01, 0.11]) * 1e-4  # exciton-exciton annihilation rate

# SIMULATION PARAMETERS
# exciton density simulation
time_range = (0, 101e-11)  # time range
diff_solver_resolution = int(1e6)  # resolution
pulse_energies = np.array([1.62, 1.62]) * e  # energy of first and second pulse
pulse_width = 1e-20  # pulse width of delta approximation
pulse_height = 1  # pulse height of delta approximation (OBSOLETE, set to 1 to use N0 as initial density)
delta_t = 100e-12  # pulse delay of example output
laser_r = np.array([0.4, 0.4]) * 1e-6  # illuminated radii of first and second pulse
laser_f = np.array([80e6, 80e6])  # frequencies of first and second laser pulse
laser_p = np.array([1., 1.])  # time-averaged laser powers of first and second pulse
# if needed, set plot limits. set to None for autoscale
plot_t_lims = None  # (0e-12, 5e-12)
plot_n_lims = None  # (1e18, 4e18)
plot_pc_lims = None  # (-0e21, 1.9e21)

# time-resolved photocurrent simulation
delta_t_sweep = (-10e-11, 5e-12)  # delta_t range
delta_t_resolution = int(1e1 + 1)  # resolution
extractions = np.array([0., 1])  # exciton extraction factors


########################################################################################################################
def param_updater():
    global number_of_tasks, alpha, gamma, tau, pulse_energies, laser_p, time_range, diff_solver_resolution, \
        delta_t_sweep, delta_t_resolution, save_index, folder_name
    """
    Reads arguments passed to the script in this order:
    NUMBER OF TASKS, ALPHA_0, ALPHA_1, GAMMA_0, GAMMA_1, TAU_0, TAU_1, ENERGY_0, ENERGY_1, POWER_0, POWER_1,
    TIME_RANGE, TIME_RESOLUTION, DELTA_T_SWEEP, DELTA_T_RESOLUTION, SAVE_INDEX, SWEEP_MODE
    And updates the parameters.
    """

    if len(sys.argv) > 1:
        number_of_tasks = int(sys.argv[1])
        alpha = np.array([sys.argv[2], sys.argv[3]], dtype=float)
        gamma = np.array([sys.argv[4], sys.argv[5]], dtype=float)
        tau = np.array([sys.argv[6], sys.argv[7]], dtype=float)
        pulse_energies = np.array([sys.argv[8], sys.argv[9]], dtype=float)
        laser_p = np.array([sys.argv[10], sys.argv[11]], dtype=float)
        time_range = (float(sys.argv[12]), float(sys.argv[13]))
        diff_solver_resolution = int(sys.argv[14])
        delta_t_sweep = (float(sys.argv[15]), float(sys.argv[16]))
        delta_t_resolution = int(sys.argv[17])
        save_index = int(sys.argv[18])
        folder_name = str(sys.argv[19])
        # os.system("cmd /k echo I am working on it!")


param_updater()

time_vals = np.linspace(time_range[0], time_range[1], diff_solver_resolution)
delta_t_step = int(delta_t / ((time_range[1] - time_range[0]) / diff_solver_resolution))
if np.sign(delta_t_sweep[0]) != np.sign(delta_t_sweep[1]):
    delta_t_pos_frac = max([delta_t_sweep[0], delta_t_sweep[1]]) / abs(delta_t_sweep[1] - delta_t_sweep[0])

if number_of_tasks == 'auto':
    number_of_tasks = mp.cpu_count()

# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mos2 = np.loadtxt(".\\extinction_mos2.txt", delimiter=',') * 1e-6
# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mose2 = np.loadtxt(".\\extinction_mose2.txt", delimiter=',') * 1e-6

powers = np.array([laser_p, laser_p])
N0_init = get_absorption_coefficients(pulse_energies / e, extinction_data_mos2, extinction_data_mose2) / (
        laser_f * np.pi * np.square(laser_r) * pulse_energies) * powers  # initial exciton densities

single_pulse_neg = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0_init[:, 1],
                                              params=(alpha, tau, gamma))
single_pulse_pos = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0_init[:, 0],
                                              params=(alpha, tau, gamma))
single_pulse_chopper = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0_init[:, 0],
                                                  params=(alpha, tau, gamma))
_, double_pulse_vals_neg = two_pulses_photoresponse(t_eval=time_vals, delta_t_steps=delta_t_step,
                                                    N_first_pulse=single_pulse_neg,
                                                    params=(alpha, tau, gamma),
                                                    N0=N0_init, res=diff_solver_resolution, negswitch=True)
_, double_pulse_vals_pos = two_pulses_photoresponse(t_eval=time_vals, delta_t_steps=delta_t_step,
                                                    N_first_pulse=single_pulse_pos,
                                                    params=(alpha, tau, gamma),
                                                    N0=N0_init, res=diff_solver_resolution, negswitch=False)

if __name__ != '__main__':
    if np.sign(delta_t_sweep[0]) != np.sign(delta_t_sweep[1]):
        pbar = tqdm(total=delta_t_resolution)
    else:
        pbar = tqdm(total=delta_t_resolution)


def pool_wrapper(dt_index):
    pbar.update(number_of_tasks)
    return photocurrent_delta_t_discrete_index(dt_index[0], single_pulse_neg, single_pulse_chopper, t_eval=time_vals,
                                               a_fac=extractions,
                                               params=(alpha, tau, gamma), N0=N0_init, res=diff_solver_resolution,
                                               negswitch=bool(dt_index[1]))


def pool_manager(instances, trange, tstep):
    """
    Creates multiple processes through pool object and manages i/o streams.

    :param instances: number of parallel processes
    :type instances: int
    :param trange: time indices to be evaluated and positive (0) or negative (1) range
    :type trange: 2d-array
    :param tstep: real time step of trange
    :type tstep: float
    :return: time-dependent photocurrent
    :rtype: nd-array
    """
    pool = mp.Pool(instances)
    start = datetime.utcnow()

    raw_result = np.array(pool.map(pool_wrapper, trange))

    result_pos = raw_result[trange[:, 1] == 0]
    result_neg = raw_result[trange[:, 1] == 1]
    result_neg[:, 0] *= -1
    result_neg = result_neg[::-1]
    result = np.vstack((result_neg, result_pos))

    print(f'\n\nINFO: Simulation took {datetime.utcnow() - start}')
    result[:, 0] *= tstep

    return result


def create_trange_array(resolution, sweeprange):
    """
    Creates an evenly spaced two dimensional time array carrying the index of time steps (dim 1) and whether the
    time is negative (1) or positive (0) in second dimension.

    :param resolution: number of time steps
    :type resolution: int
    :param sweeprange: minimal and maximal time values
    :type sweeprange: tuple
    :return: 2d-time array with indices and step size in real time
    :rtype: nd-array, float
    """
    tstep = abs(time_range[1] - time_range[0]) / len(time_vals)
    trange = np.ones((resolution, 2), dtype=int)

    if np.sign(sweeprange[0]) != np.sign(sweeprange[1]):
        delta_t_resolution_pos = int(delta_t_resolution * delta_t_pos_frac)
        delta_t_resolution_neg = delta_t_resolution - delta_t_resolution_pos
        trange[:delta_t_resolution_neg, 0] = np.linspace(1, int(abs(sweeprange[0]) / tstep), delta_t_resolution_neg)
        trange[delta_t_resolution_neg:, 0] = np.linspace(1, int(abs(sweeprange[1]) / tstep), delta_t_resolution_pos)
        trange[delta_t_resolution_neg:, 1] *= 0
        return trange, tstep

    if abs(sweeprange[0] / tstep) < 0.5:
        start = 1
    else:
        start = int(abs(sweeprange[0]) / tstep)
        print(start)

    if np.sign(sweeprange[0]) == -1:
        trange[:, 0] = np.linspace(start, int(abs(sweeprange[1]) / tstep), resolution)

    else:
        trange[:, 0] = np.linspace(start, int(sweeprange[1] / tstep), resolution)
        trange[:, 1] *= 0

    return trange, tstep


def data_saver(filepath, result_data, paramsave=True, print_out=False, folder_name=folder_name):
    """
    Saves photocurrent and optianlly parameters into file. Creates new folder from timestamp by default.

    :param filepath: path of main output directory
    :type filepath: str
    :param result_data: pc data
    :type result_data: 2d-array
    :param paramsave: toggle parameter saving
    :type paramsave: bool
    :param print_out: toggle confirmation message
    :type print_out: bool
    :param folder_name: if provided, omits automatic timestamp folder and saves to folder_name
    :type folder_name: str
    """

    rframe = pd.DataFrame({'delta_t': result_data[:, 0], 'pc': result_data[:, 1]})
    pframe = pd.DataFrame({'alpha': alpha, 'gamma': gamma, 'tau': tau, 'energies': pulse_energies, 'powers': laser_p,
                           'extractions': extractions, 'n_time_range': time_range,
                           'n_resolution': diff_solver_resolution, 'delta_t_range': delta_t_sweep,
                           'delta_t_resolution': delta_t_resolution})

    if use_date_as_folder_name:
        now = timestamp_maker()
        filepath = filepath + f'\\{now}\\'
    else:
        filepath = filepath + f'\\{folder_name}\\'

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    filepath_d = f'{filepath}data_' + '{:04d}'.format(save_index) + '.txt'
    filepath_p = f'{filepath}params_' + '{:04d}'.format(save_index) + '.txt'
    rframe.to_csv(filepath_d, sep=';', index=False)
    if paramsave:
        pframe.to_csv(filepath_p, sep=';', index=False)
    if print_out:
        print(f'Saved to {filepath}')


if __name__ == '__main__':
    # CHECKS
    if delta_t_sweep[1] >= time_range[1] / 2:
        print(
            'WARNING: Simulation range should be at least two times the maximum pulse delay. This is not the case with '
            f'dt = {delta_t_sweep[1]} and tmax = {time_range[1]} !')
    if delta_t_resolution % 2 == 0:
        print('WARNING: It is highly recommended to use an odd resolution for sweeping pulse delay !')
    if (time_range[1] - time_range[0]) / diff_solver_resolution >= abs(
            delta_t_sweep[1] - delta_t_sweep[0]) / delta_t_resolution * 1e-2:
        print(
            f'WARNING: dt = {(delta_t_sweep[1] - delta_t_sweep[0]) / delta_t_resolution} is too small to be resolved properly at a resolution of '
            f'{(time_range[1] - time_range[0]) / diff_solver_resolution} !')

    print(f'INFO: {datetime.now()}: Starting simulation...\n')

    delta_range, tstep = create_trange_array(delta_t_resolution, delta_t_sweep)

    pc = pool_manager(number_of_tasks, trange=delta_range, tstep=tstep)

    if save_data:
        data_saver(savedir, pc)

    float_formatter = '{:.4e}'.format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    if pc_plot:
        fig, ax = plt.subplots()
        ax.plot(pc[:, 0], pc[:, 1], 'x-')
        ax.set_xlabel('$\\Delta t$')
        ax.set_ylabel('$PC(\\Delta t)$')
        ax.set_title(f'nt={time_range}, nres={diff_solver_resolution}, dtres={delta_t_resolution}')
        textstr = f'$\\alpha$={alpha} \n' \
                  f'$\\tau$={tau} \n' \
                  f'$\\gamma$={gamma} \n' \
                  f'$N_0$={N0_init} \n' \
                  f'$E_p$={pulse_energies / e}\n' \
                  f'$a$={extractions}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
        ax.set_ylim(plot_pc_lims)

        np.savetxt(f'(E_p={pulse_energies / e},tau={tau},gamma={gamma},alpha={alpha}).txt', pc, header=textstr)

    if logscale:
        # ax.set_xscale('log')
        ax.set_yscale('log')

    if example_output:
        fig1, ax1 = plt.subplots()
        ax1.plot(double_pulse_vals_pos[2], double_pulse_vals_pos[1], '-')
        ax1.plot(double_pulse_vals_neg[2], double_pulse_vals_neg[1], '-')
        ax1.set_xlabel('$t$')
        ax1.set_ylabel('$N(t)$')
        ax1.set_title(f'resolution={diff_solver_resolution}')
        textstr = f'$\\alpha$={alpha} \n' \
                  f'$\\tau$={tau} \n' \
                  f'$\\gamma$={gamma} \n' \
                  f'$N_0$={N0_init} \n' \
                  f'$E_p$={pulse_energies / e} \n' \
                  f'$a$={extractions}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.6, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
        ax1.set_xlim(plot_t_lims)
        ax1.set_ylim(plot_n_lims)

    if plot_extinction:
        fig2, ax2 = plt.subplots()
        ax2.plot(wavelength_to_energy(extinction_data_mos2[:, 0]), extinction_data_mos2[:, 1], label='MoS2')
        ax2.plot(wavelength_to_energy(extinction_data_mose2[:, 0]), extinction_data_mose2[:, 1], label='MoSe2')
        ax2.set_xlabel('energy (eV)')
        ax2.set_ylabel('extinction coefficent')
        ax2.legend()

    plt.show()
