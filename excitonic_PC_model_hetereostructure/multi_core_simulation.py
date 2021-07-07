import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
from parallelization_framework import get_absorption_coefficients, single_pulse_photoresponse, \
    two_pulses_photoresponse, photocurrent_delta_t_discrete_index, wavelength_to_energy

# # NATURAL CONSTANTS
h = 6.62607015e-34
c = 2.99792458e8
e = 1.602176634e-19

########################################################################################################################
# https://www.desmos.com/calculator/mqnypi6npa
# MAIN SIMULATION TOGGLE
example_output = True
plot_extinction = False
number_of_tasks = 'auto'
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
delta_t_resolution = int(5e1 + 1)  # resolution
extractions = np.array([0., 1])  # exciton extraction factors

########################################################################################################################
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
