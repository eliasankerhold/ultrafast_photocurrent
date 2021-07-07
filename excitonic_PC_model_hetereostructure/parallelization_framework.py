import numpy as np
from scipy.integrate import solve_ivp, trapz
from bisect import bisect_left

# # NATURAL CONSTANTS
h = 6.62607015e-34
c = 2.99792458e8
e = 1.602176634e-19


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


def get_absorption_coefficients(energies, data_mos2, data_mose2):
    """
    Calculates absorption coefficients for both pulse energies.

    :param data_mos2: extinction data for mos2
    :type data_mos2: 1d-array
    :param data_mose2: extinction data for mose2
    :type data_mose2: 1d-array
    :param energies: photon energies of first and second pulse in J
    :type energies: 1d-array
    :return: absorption coefficients for both materials and both pulses (material, pulse_number)
    :rtype: (2, 2) array
    """
    absorption = np.zeros((2, 2))
    lambda_onetwo = energy_to_wavelength(energies)
    if (data_mos2[:, 0].min() < lambda_onetwo[0] < data_mos2[:, 0].max()) and \
            (data_mose2[:, 0].min() < lambda_onetwo[0] < data_mose2[:, 0].max()):
        if (data_mos2[:, 0].min() < lambda_onetwo[1] < data_mos2[:, 0].max()) and \
                (data_mose2[:, 0].min() < lambda_onetwo[1] < data_mose2[:, 0].max()):
            k_mos2 = np.array(
                [take_closest(data_mos2[:, 0], lambda_onetwo[0]),
                 take_closest(data_mos2[:, 0], lambda_onetwo[1])])
            k_mose2 = np.array(
                [take_closest(data_mose2[:, 0], lambda_onetwo[0]),
                 take_closest(data_mose2[:, 0], lambda_onetwo[1])])
            absorption[0, :] = 4 * np.pi * data_mos2[k_mos2, 1]
            absorption[1, :] = 4 * np.pi * data_mose2[k_mose2, 1]
    else:
        print(
            f'ERROR: No extinction coefficient data for wavelengths {lambda_onetwo} m corresponding to pulse energies '
            f'{energies * e} eV available !')

    return absorption


def delta_approx(x, smearing=0, p_height=1):
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
    :param N_init: inital exciton densities
    :type N_init: 1d-array
    :return: excitation factors for both monolayers
    :rtype: 2d-array
    """
    g1 = N_init[0] * delta_approx(t, smearing=smearing)
    g2 = N_init[1] * delta_approx(t, smearing=smearing)
    return np.array([g1, g2])


def rhs_coupled_photoresponse(t, Nvec, N_init, params, t_shift=0):
    """
    Right hand side of the non-linear coupled photoresponse model.

    :param N_init: initial exciton density
    :type N_init: 1d-array
    :param params: parameters alpha, tau, gamma
    :type params: tuple
    :param t_shift:
    :type t_shift:
    :param t: time
    :type t: float
    :param Nvec: value of exciton function density
    :type Nvec: 2d-array
    :return: left hand side
    :rtype: 2d-array
    """
    alpha, tau, gamma = params
    Gt = excitation_function(t + t_shift, smearing=1, N_init=N_init)
    dN1 = Gt[0] - Nvec[0] / tau[0] - gamma[0] * Nvec[0] ** 2 + alpha[0] * Nvec[1] - alpha[1] * Nvec[0]
    dN2 = Gt[1] - Nvec[1] / tau[1] - gamma[1] * Nvec[1] ** 2 + alpha[1] * Nvec[0] - alpha[0] * Nvec[1]
    return np.array([dN1, dN2])


def single_pulse_photoresponse(t_span, t_eval, N_init, params):
    """
    Solves the system for a single pulse at t=0.

    :param params: parameters alpha, tau, gamma
    :type params: tuple
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
                         vectorized=True, dense_output=True, args=(N_init, params,))
    return solution


def two_pulses_photoresponse(t_eval, delta_t_steps, N_first_pulse, params, N0, res, negswitch=False):
    """
    Solves the system for two pulses with a difference in time.

    :param params: parameters alpha, tau, gamma
    :type params: tuple
    :param N0: initial exciton densities
    :type N0: 1d-array
    :param res: resolution of exciton density solution
    :type res: int
    :param negswitch: toggle negative pulse delay mode
    :type negswitch: bool
    :param t_eval: time points at which function values are saved
    :type t_eval: 1d-array
    :param delta_t_steps: time difference in steps of the total time range array
    :type delta_t_steps: int
    :param N_first_pulse: solution of the system for one pulse
    :type N_first_pulse: scipy solution object
    :return: solved system
    :rtype: scipy solution object, 2d-array
    """
    N0 = N0
    narray = np.zeros((3, len(t_eval)))
    narray[:2, :delta_t_steps] = N_first_pulse.sol(t_eval[:delta_t_steps])
    if negswitch:
        N0_delta_t = N0[:, 0] + N_first_pulse.sol(t_eval[delta_t_steps])
    else:
        N0_delta_t = N0[:, 1] + N_first_pulse.sol(t_eval[delta_t_steps])
    nsol = solve_ivp(rhs_coupled_photoresponse, t_span=(t_eval[delta_t_steps], t_eval[-1]),
                     t_eval=np.linspace(t_eval[delta_t_steps], t_eval[-1], res),
                     y0=N0_delta_t, method='RK45', vectorized=True, rtol=0.01, dense_output=True,
                     args=(N0_delta_t, params, -t_eval[delta_t_steps],))
    narray[:2, delta_t_steps:] = nsol.sol(t_eval[delta_t_steps:])
    narray[2, :] = t_eval

    return nsol, narray


def lock_in_photocurrent_discrete(delta_t, single_pulse, single_pulse_chopper, t_eval, a_fac, params, N0, res,
                                  negswitch=False):
    """
    Evaluates the delta_t - dependent PC integral using trapezoid rule.

    :param params: parameters alpha, tau, gamma
    :type params: tuple
    :param N0: initial exciton densities
    :type N0: 1d-array
    :param res: resolution of exciton density solution
    :type res: int
    :param negswitch: toggle negative pulse delay mode
    :type negswitch: bool
    :param delta_t: pulse delay index
    :type delta_t: int
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
    n2p, n2p_vals = two_pulses_photoresponse(t_eval, delta_t, single_pulse, params, N0, res, negswitch=negswitch)
    integrand1 = a_fac[0] * n2p_vals[0] + a_fac[1] * n2p_vals[1]
    integrand2 = -  a_fac[0] * single_pulse_chopper.y[0] - a_fac[1] * single_pulse_chopper.y[1]
    pc = trapz(integrand1) - trapz(integrand2)

    return pc


def photocurrent_delta_t_discrete(delta_t_range, pc_resolution, single_pulse, single_pulse_chopper, t_span, t_eval,
                                  a_fac, params, N0, res,
                                  negswitch=False):
    """
    Calculates the photocurrent in a given range of delta_t.

    :param params: parameters alpha, tau, gamma
    :type params: tuple
    :param N0: initial exciton densities
    :type N0: 1d-array
    :param res: resolution of exciton density solution
    :type res: int
    :param negswitch: toggle negative pulse delay mode
    :type negswitch: bool
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
        pc_vals[i] = lock_in_photocurrent_discrete(dt, single_pulse, single_pulse_chopper, t_eval, a_fac, params, N0,
                                                   res,
                                                   negswitch=negswitch)

    return trange * tstep, pc_vals


def photocurrent_delta_t_discrete_index(delta_t_index, single_pulse, single_pulse_chopper, t_eval, a_fac, params, N0,
                                        res, negswitch=False):
    """
    Calculates the photocurrent in a given range of delta_t.

    :param delta_t_index: index of pulse delay value with respect to exciton density time values
    :type delta_t_index: int
    :param params: parameters alpha, tau, gamma
    :type params: tuple
    :param N0: initial exciton densities
    :type N0: 1d-array
    :param res: resolution of exciton density solution
    :type res: int
    :param negswitch: toggle negative pulse delay mode
    :type negswitch: bool
    :param single_pulse: solution of the first pulse
    :type single_pulse: scipy solution object
    :param t_eval: time points at which the exciton density functions are evaluated
    :type t_eval: 1d-array
    :param a_fac: extraction factors
    :type a_fac: 1d-array
    :param E2: photon energy of the second pulse
    :type E2: float
    :return: delta_t values and corresponding PC values
    :rtype: 1d-array, 1d-array
    """
    if delta_t_index == 0:
        return None, None

    pc_val = lock_in_photocurrent_discrete(delta_t_index, single_pulse, single_pulse_chopper, t_eval, a_fac, params, N0,
                                           res,
                                           negswitch=negswitch)

    return delta_t_index, pc_val
