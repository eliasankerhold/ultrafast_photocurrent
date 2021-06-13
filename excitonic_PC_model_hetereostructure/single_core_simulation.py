import numpy as np
from scipy.integrate import solve_ivp, trapz
import matplotlib.pyplot as plt
from bisect import bisect_left
from datetime import datetime

# NATURAL CONSTANTS
h = 6.62607015e-34
c = 2.99792458e8
e = 1.602176634e-19

########################################################################################################################
# https://www.desmos.com/calculator/mqnypi6npa
# MAIN SIMULATION TOGGLE
run_main = True
individual_output = []  # list(np.array([0, 2, 4, 6, 8, 10]) * 1)  # list(np.arange(0, 100, 5))
example_output = False
plot_extinction = True
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
delta_t_sweep = (0e-12, 200e-12)  # delta_t range
delta_t_resolution = int(1e1 + 1)  # resolution
extractions = np.array([1., 1])  # exciton extraction factors

########################################################################################################################
time_vals = np.linspace(time_range[0], time_range[1], diff_solver_resolution)
delta_t_step = int(delta_t / ((time_range[1] - time_range[0]) / diff_solver_resolution))
history = []

# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mos2 = np.loadtxt(".\\extinction_mos2.txt", delimiter=',') * 1e-6
# Hsu et al. 2019: monolayer (1L) film; n,k 0.40-0.86 µm
extinction_data_mose2 = np.loadtxt(".\\extinction_mose2.txt", delimiter=',') * 1e-6

# CHECKS
if delta_t_sweep[1] >= time_range[1] / 2:
    print('WARNING: Simulation range should be at least two times the maximum pulse delay. This is not the case with '
          f'dt = {delta_t_sweep[1]} and tmax = {time_range[1]} !')
if delta_t_resolution % 2 == 0:
    print('WARNING: It is highly recommended to use an odd resolution for sweeping pulse delay !')
if (time_range[1] - time_range[0]) / diff_solver_resolution >= abs(
        delta_t_sweep[1] - delta_t_sweep[0]) / delta_t_resolution * 1e-2:
    print(f'WARNING: dt = {delta_t} is too small to be resolved properly at a resolution of '
          f'{(time_range[1] - time_range[0]) / diff_solver_resolution} !')

hist = []


def energy_to_wavelength(E):
    """
    Converts energy (J) into corresponding wavelength (m) .

    :param E: energy in J
    :type E: 1d-array
    :return: wavelength in m
    :rtype: 1d-array
    """
    return (h * c) / (E * e)


def wavelength_to_energy(lam):
    """
    Converts wavelength (m) into energy (J).

    :param lam: wavelength in m
    :type lam: 1d-array
    :return: energy in J
    :rtype: 1d-array
    """
    return (h * c) / (lam * e)


def take_closest(search_list, target_value):
    """
    Assumes search_list is sorted. Returns index of closest value to target_value.
    If two numbers are equally close, return the index of smallest number.

    :param search_list: list to be searched
    :type search_list: list
    :param target_value: value to be searched for
    :type target_value: float
    :return: index of closest entry in search_list
    :rtype: int
    """
    pos = bisect_left(search_list, target_value)
    if pos == 0:
        return 0
    if pos == len(search_list):
        return -1
    before = search_list[pos - 1]
    after = search_list[pos]
    if after - target_value < target_value - before:
        return pos
    else:
        return pos - 1


def get_absorption_coefficients(energies):
    """
    Calculates absorption coefficients for both pulse energies.

    :param energies: photon energies of first and second pulse in J
    :type energies: 1d-array
    :return: absorption coefficients for both materials and both pulses (material, pulse_number)
    :rtype: (2, 2) array
    """
    absorption = np.zeros((2, 2))
    lambda_onetwo = energy_to_wavelength(energies)
    if (extinction_data_mos2[:, 0].min() < lambda_onetwo[0] < extinction_data_mos2[:, 0].max()) and \
            (extinction_data_mose2[:, 0].min() < lambda_onetwo[0] < extinction_data_mose2[:, 0].max()):
        if (extinction_data_mos2[:, 0].min() < lambda_onetwo[1] < extinction_data_mos2[:, 0].max()) and \
                (extinction_data_mose2[:, 0].min() < lambda_onetwo[1] < extinction_data_mose2[:, 0].max()):
            k_mos2 = np.array(
                [take_closest(extinction_data_mos2[:, 0], lambda_onetwo[0]),
                 take_closest(extinction_data_mos2[:, 0], lambda_onetwo[1])])
            k_mose2 = np.array(
                [take_closest(extinction_data_mose2[:, 0], lambda_onetwo[0]),
                 take_closest(extinction_data_mose2[:, 0], lambda_onetwo[1])])
            absorption[0, :] = 4 * np.pi * extinction_data_mos2[k_mos2, 1]
            absorption[1, :] = 4 * np.pi * extinction_data_mose2[k_mose2, 1]
    else:
        print(
            f'ERROR: No extinction coefficient data for wavelengths {lambda_onetwo} m corresponding to pulse energies '
            f'{energies * e} eV available !')

    return absorption


def delta_approx(x, smearing=0, p_height=pulse_height):
    """
    Approximates delta function as rectangular pulse.

    :param p_height: pulse height
    :type p_height: float
    :param x: one-dimensional variable
    :type x: float
    :param smearing: pulsewidth
    :type smearing: float
    :return: evaluated delta function
    :rtype: float
    """
    if -smearing < x < smearing:
        return p_height
    else:
        return 0


def excitation_function(t, smearing, N_init):
    """
    Model excitation pulse as delta function for both monolayers.

    :param t: time
    :type t: float
    :param smearing: pulsewidth of delta approximation
    :type smearing: float
    :return: excitation factors for both monolayers
    :rtype: 2d-array
    """
    g1 = N_init[0] * delta_approx(t, smearing=smearing)
    g2 = N_init[1] * delta_approx(t, smearing=smearing)
    return np.array([g1, g2])


def rhs_coupled_photoresponse(t, Nvec, N_init, t_shift=0):
    """
    Right hand side of the non-linear coupled photoresponse model.

    :param t: time
    :type t: float
    :param Nvec: value of exciton function density
    :type Nvec: 2d-array
    :return: left hand side
    :rtype: 2d-array
    """
    Gt = excitation_function(t + t_shift, smearing=pulse_width, N_init=N_init)
    dN1 = Gt[0] - Nvec[0] / tau[0] - gamma[0] * Nvec[0] ** 2 + alpha[0] * Nvec[1]
    dN2 = Gt[1] - Nvec[1] / tau[1] - gamma[1] * Nvec[1] ** 2 + alpha[1] * Nvec[0]
    hist.append([dN1, dN2])
    return np.array([dN1, dN2])


def single_pulse_photoresponse(t_span, t_eval, N_init):
    """
    Solves the system for a single pulse at t=0.

    :param t_span: timespan of simulation
    :type t_span: tuple
    :param t_eval: time points at which function values are saved
    :type t_eval: 1d-array
    :param N_init: initial exciton densities
    :type N_init: 2d-array
    :return: solved exciton functions
    :rtype: scipy solution object
    """
    solution = solve_ivp(rhs_coupled_photoresponse, t_span=t_span, t_eval=t_eval, y0=N_init, method='RK45',
                         vectorized=True, dense_output=True, args=(N_init,))
    return solution


def two_pulses_photoresponse(t_eval, delta_t_steps, N_first_pulse, negswitch=False):
    """
    Solves the system for two pulses with a difference in time.

    :param t_eval: time points at which function values are saved
    :type t_eval: 1d-array
    :param delta_t_steps: time difference in steps of the total time range array
    :type delta_t_steps: int
    :param N_first_pulse: solution of the system for one pulse
    :type N_first_pulse: scipy solution object
    :return: solved system
    :rtype: scipy solution object, 2d-array
    """
    narray = np.zeros((3, len(t_eval)))
    narray[:2, :delta_t_steps] = N_first_pulse.sol(t_eval[:delta_t_steps])
    if negswitch:
        N0_delta_t = N0[:, 0] + N_first_pulse.sol(t_eval[delta_t_steps])
    else:
        N0_delta_t = N0[:, 1] + N_first_pulse.sol(t_eval[delta_t_steps])
    nsol = solve_ivp(rhs_coupled_photoresponse, t_span=(t_eval[delta_t_steps], t_eval[-1]),
                     t_eval=np.linspace(t_eval[delta_t_steps], t_eval[-1], diff_solver_resolution),
                     y0=N0_delta_t, method='RK45', vectorized=True, rtol=0.01, dense_output=True,
                     args=(N0_delta_t, -t_eval[delta_t_steps],))
    narray[:2, delta_t_steps:] = nsol.sol(t_eval[delta_t_steps:])
    narray[2, :] = t_eval

    return nsol, narray


def lock_in_photocurrent_discrete(delta_t, single_pulse, t_eval, a_fac, negswitch=False):
    """
    Evaluates the delta_t - dependent PC integral using trapezoid rule.

    :param delta_t: pulse delay
    :type delta_t: float
    :param single_pulse: solution of the first pulse
    :type single_pulse: scipy solution object
    :param t_eval: times at which the exciton density should be evaluated
    :type t_eval: 1d-array
    :param a_fac: extraction factors
    :type a_fac: 1d-array
    :param E2: photon energy of the second pulse
    :type E2: float
    :return: photocurrent
    :rtype: float
    """
    n2p, n2p_vals = two_pulses_photoresponse(t_eval, delta_t, single_pulse, negswitch=negswitch)
    if negswitch:
        delta_t *= -1
    history.append([n2p, n2p_vals, delta_t])
    integrand = a_fac[0] * (n2p_vals[0] - single_pulse.y[0]) + a_fac[1] * (n2p_vals[1] - single_pulse.y[1])

    pc = trapz(integrand)

    return pc


def photocurrent_delta_t_discrete(delta_t_range, pc_resolution, single_pulse, t_span, t_eval, a_fac, negswitch=False):
    """
    Calculates the photocurrent in a given range of delta_t.

    :param delta_t_range: minimal and maximal delta_t
    :type delta_t_range: tuple
    :param pc_resolution: number of points to evaluate
    :type pc_resolution: int
    :param single_pulse: solution of the first pulse
    :type single_pulse: scipy solution object
    :param t_span: time range in which the exciton density functions are evaluated
    :type t_span: tuple
    :param t_eval: time points at which the exciton density functions are evaluated
    :type t_eval: 1d-array
    :param a_fac: extraction factors
    :type a_fac: 1d-array
    :param E2: photon energy of the second pulse
    :type E2: float
    :return: delta_t values and corresponding PC values
    :rtype: 1d-array, 1d-array
    """
    myname = 'photocurrent_delta_t_discrete'
    tstep = (t_span[1] - t_span[0]) / len(t_eval)
    if abs(delta_t_range[0] / tstep) < 0.5:
        print(f'\nINFO: {myname}: Set start to 1!')
        start = 1
    else:
        start = int(delta_t_range[0] / tstep)
    trange = np.linspace(start, int(delta_t_range[1] / tstep), pc_resolution, dtype=int)
    pc_vals = np.zeros_like(trange, dtype=float)
    for i, dt in enumerate(trange):
        if dt == 0:
            print(f'\nINFO: {myname}: skipping dt = 0!')
            continue
        print(f'\r{i / len(trange) * negative_flag * 100 + negative_offset:2.2f} %', end='')
        pc_vals[i] = lock_in_photocurrent_discrete(dt, single_pulse, t_eval, a_fac, negswitch=negswitch)

    return trange * tstep, pc_vals


if run_main:
    start = datetime.utcnow()
    print(f'INFO: {datetime.now()}: Starting simulation. ')

    powers = np.array([laser_p, laser_p])
    N0 = get_absorption_coefficients(pulse_energies / e) / (
            laser_f * np.pi * np.square(laser_r) * pulse_energies) * powers  # initial exciton densities
    single_pulse = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0[:, 0])
    double_pulse, double_pulse_vals = two_pulses_photoresponse(t_eval=time_vals, delta_t_steps=delta_t_step,
                                                               N_first_pulse=single_pulse)

    if np.sign(delta_t_sweep[0]) == -1:
        negative_flag, negative_offset = 0.5, 0
        delta_t_sweep_neg = (0, delta_t_sweep[0] * -1)
        delta_t_sweep_pos = (0, delta_t_sweep[1])
        single_pulse_pos = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0[:, 0])
        single_pulse_neg = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0[:, 1])

        pc_sim_neg = photocurrent_delta_t_discrete(delta_t_sweep_neg, pc_resolution=int(delta_t_resolution / 2),
                                                   single_pulse=single_pulse, t_span=time_range, t_eval=time_vals,
                                                   a_fac=extractions, negswitch=True)
        negative_offset = 50
        pc_sim_pos = photocurrent_delta_t_discrete(delta_t_sweep_pos, pc_resolution=int(delta_t_resolution / 2),
                                                   single_pulse=single_pulse, t_span=time_range, t_eval=time_vals,
                                                   a_fac=extractions, negswitch=False)

        pc_sim_neg = ((pc_sim_neg[0] * -1)[::-1], pc_sim_neg[1][::-1])
        pc_sim = np.hstack((pc_sim_neg, pc_sim_pos))

    else:
        negative_flag, negative_offset = 1, 0
        single_pulse = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0[:, 0])
        pc_sim = photocurrent_delta_t_discrete(delta_t_sweep, pc_resolution=delta_t_resolution,
                                               single_pulse=single_pulse, t_span=time_range, t_eval=time_vals,
                                               a_fac=extractions)

    first_excitation = np.array(
        [excitation_function(t, smearing=pulse_width, N_init=N0[:, 0]) for t in time_vals])
    second_excitation = np.array(
        [excitation_function(t - time_vals[delta_t_step], smearing=pulse_width, N_init=N0[:, 1])
         for t in time_vals])

    end = datetime.utcnow()
    print(f'\nINFO: Simulations took {end - start} to compute (without plotting).')

    float_formatter = '{:.4e}'.format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    fig, ax = plt.subplots()
    ax.plot(pc_sim[0], pc_sim[1], 'x-')
    ax.set_xlabel('$\\Delta t$')
    ax.set_ylabel('$PC(\\Delta t)$')
    ax.set_title(f'nt={time_range}, nres={diff_solver_resolution}, dtres={delta_t_resolution}')
    textstr = f'$\\alpha$={alpha} \n' \
              f'$\\tau$={tau} \n' \
              f'$\\gamma$={gamma} \n' \
              f'$N_0$={N0} \n' \
              f'$E_p$={pulse_energies / e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
    ax.set_ylim(plot_pc_lims)

    if individual_output:
        for i in individual_output:
            h = history[i]
            fign, axn = plt.subplots()
            axn.plot(h[1][2], h[1][1])
            axn.set_title(str(h[2] / delta_t_step))
            axn.set_xlim(plot_t_lims)
            axn.hlines([N0[0], N0[1]], h[1][2].min(), h[1][1].max(), color='gray')

    if example_output:
        fig1, ax1 = plt.subplots()
        ax1.plot(double_pulse_vals[2], double_pulse_vals[1], '-')
        ax1.set_xlabel('$t$')
        ax1.set_ylabel('$N(t)$')
        ax1.set_title(f'resolution={diff_solver_resolution}')
        textstr = f'$\\alpha$={alpha} \n' \
                  f'$\\tau$={tau} \n' \
                  f'$\\gamma$={gamma} \n' \
                  f'$N_0$={N0} \n' \
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

else:
    print(get_absorption_coefficients(pulse_energies))
    plt.show()
