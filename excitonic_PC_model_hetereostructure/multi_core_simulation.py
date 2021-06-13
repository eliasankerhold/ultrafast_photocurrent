from parallelization_framework import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter.ttk import Progressbar
from time import sleep
from tqdm import tqdm

########################################################################################################################
# https://www.desmos.com/calculator/mqnypi6npa
# MAIN SIMULATION TOGGLE
example_output = False
plot_extinction = False
number_of_tasks = 'auto'
# DEVICE PARAMETERS - first entry is MoSe2 (top), second is MoS2 (bottom)
alpha = np.array([0e11, 0e11])  # coupling constants
tau = np.array([25, 25]) * 1e-12  # photoresponse time
gamma = np.array([1, 1]) * 1e-4  # exciton-exciton annihilation rate

# SIMULATION PARAMETERS
# exciton density simulation
time_range = (0, 101e-11)  # time range
diff_solver_resolution = int(1e6)  # resolution
pulse_energies = np.array([1.6, 1.6]) * e  # energy of first and second pulse
pulse_width = 1e-20  # pulse width of delta approximation
pulse_height = 1  # pulse height of delta approximation (OBSOLETE, set to 1 to use N0 as initial density)
delta_t = 1e-12  # pulse delay of example output
laser_r = np.array([0.4, 0.4]) * 1e-6  # illuminated radii of first and second pulse
laser_f = np.array([80e6, 80e6])  # frequencies of first and second laser pulse
laser_p = np.array([1., 1.])  # time-averaged laser powers of first and second pulse
# if needed, set plot limits. set to None for autoscale
plot_t_lims = None  # (0e-12, 5e-12)
plot_n_lims = None  # (1e18, 4e18)
plot_pc_lims = None  # (-0e21, 1.9e21)

# time-resolved photocurrent simulation
delta_t_sweep = (000e-12, 200e-12)  # delta_t range
delta_t_resolution = int(5e2 + 1)  # resolution
extractions = np.array([1., 1])  # exciton extraction factors

########################################################################################################################
time_vals = np.linspace(time_range[0], time_range[1], diff_solver_resolution)
delta_t_step = int(delta_t / ((time_range[1] - time_range[0]) / diff_solver_resolution))
if number_of_tasks == 'auto':
    number_of_tasks = mp.cpu_count()

# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mos2 = np.loadtxt(".\\extinction_mos2.txt", delimiter=',') * 1e-6
# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mose2 = np.loadtxt(".\\extinction_mose2.txt", delimiter=',') * 1e-6

powers = np.array([laser_p, laser_p])
N0_init = get_absorption_coefficients(pulse_energies / e, extinction_data_mos2, extinction_data_mose2) / (
        laser_f * np.pi * np.square(laser_r) * pulse_energies) * powers  # initial exciton densities

single_pulse = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0_init[:, 0],
                                          params=(alpha, tau, gamma))
double_pulse, double_pulse_vals = two_pulses_photoresponse(t_eval=time_vals, delta_t_steps=delta_t_step,
                                                           N_first_pulse=single_pulse, params=(alpha, tau, gamma),
                                                           N0=N0_init, res=diff_solver_resolution, negswitch=False)

if __name__ != '__main__':
    # CHECKS
    if delta_t_sweep[1] >= time_range[1] / 2:
        print(
            'WARNING: Simulation range should be at least two times the maximum pulse delay. This is not the case with '
            f'dt = {delta_t_sweep[1]} and tmax = {time_range[1]} !')
    if delta_t_resolution % 2 == 0:
        print('WARNING: It is highly recommended to use an odd resolution for sweeping pulse delay !')
    if (time_range[1] - time_range[0]) / diff_solver_resolution >= abs(
            delta_t_sweep[1] - delta_t_sweep[0]) / delta_t_resolution * 1e-2:
        print(f'WARNING: dt = {delta_t} is too small to be resolved properly at a resolution of '
              f'{(time_range[1] - time_range[0]) / diff_solver_resolution} !')

    if delta_t_sweep[0] < 0:
        pbar = tqdm(total=delta_t_resolution / 2)
    else:
        pbar = tqdm(total=delta_t_resolution)


def pool_wrapper_pos(dt_index):
    pbar.update(number_of_tasks)
    return photocurrent_delta_t_discrete_index(dt_index, single_pulse, t_eval=time_vals, a_fac=extractions,
                                               params=(alpha, tau, gamma), N0=N0_init, res=diff_solver_resolution,
                                               negswitch=False)


def pool_wrapper_neg(dt_index):
    pbar.update(number_of_tasks)
    return photocurrent_delta_t_discrete_index(dt_index, single_pulse, t_eval=time_vals, a_fac=extractions,
                                               params=(alpha, tau, gamma), N0=N0_init, res=diff_solver_resolution,
                                               negswitch=True)


def pool_manager(instances, range, negflag, doubleflag=False):
    tstep = abs(time_range[1] - time_range[0]) / len(time_vals)
    if abs(range[0] / tstep) < 0.5:
        # print('\nINFO: pool manager: Set start to 1!')
        start = 1
    else:
        start = int(range[0] / tstep)
    if negflag:
        trange = np.linspace(start, int(range[1] / tstep), int(delta_t_resolution), dtype=int)
    else:
        trange = np.linspace(start, int(range[1] / tstep), delta_t_resolution, dtype=int)
    pool = mp.Pool(instances)
    start = datetime.utcnow()
    if negflag:
        result = np.array(pool.map(pool_wrapper_neg, trange))
        result[:, 0] *= -1
        result = result[::-1]
    else:
        result = np.array(pool.map(pool_wrapper_pos, trange))
    if doubleflag:
        print('\nNegative range done.')
    else:
        print(f'\n\nINFO: Simulation took {datetime.utcnow() - start}')
    result[:, 0] *= tstep

    return result


if __name__ == '__main__':
    if np.sign(delta_t_sweep[0]) == -1:
        delta_t_resolution = int(delta_t_resolution / 2)
        delta_t_sweep_neg = (0, delta_t_sweep[0] * -1)
        delta_t_sweep_pos = (0, delta_t_sweep[1])

        pc_neg = pool_manager(number_of_tasks, delta_t_sweep_neg, negflag=True, doubleflag=True)
        pc_pos = pool_manager(number_of_tasks, delta_t_sweep_pos, negflag=False, doubleflag=False)
        pc = np.vstack((pc_neg, pc_pos))

    else:
        pc = pool_manager(number_of_tasks, delta_t_sweep, negflag=False)

    float_formatter = '{:.4e}'.format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    fig, ax = plt.subplots()
    ax.plot(pc[:, 0], pc[:, 1], 'x-')
    ax.set_xlabel('$\\Delta t$')
    ax.set_ylabel('$PC(\\Delta t)$')
    ax.set_title(f'nt={time_range}, nres={diff_solver_resolution}, dtres={delta_t_resolution}')
    textstr = f'$\\alpha$={alpha} \n' \
              f'$\\tau$={tau} \n' \
              f'$\\gamma$={gamma} \n' \
              f'$N_0$={N0_init} \n' \
              f'$E_p$={pulse_energies / e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
    ax.set_ylim(plot_pc_lims)

    if example_output:
        fig1, ax1 = plt.subplots()
        ax1.plot(double_pulse_vals[2], double_pulse_vals[1], '-')
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
