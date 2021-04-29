import numpy as np
from scipy.integrate import solve_ivp, quad, trapz
import matplotlib.pyplot as plt

########################################################################################################################
# DEVICE PARAMETERS
alpha = np.array([1, 1])  # coupling constants
tau = np.array([1, 1])  # photoresponse time
gamma = np.array([1, 1])  # exciton-exciton annihilation rate
energy_thresholds = np.array([0, 0])  # monolayer band gaps
absorption_coeff = np.array([1, 1])  # absorption coefficients of both monolayers

# SIMULATION PARAMETERS
# exciton density simulation
time_range = (0, 100)  # time range
diff_solver_resolution = int(1e5)  # resolution
pulse_energies = np.array([1, 1])  # energy of first and second pulse
pulse_width = 1e-35  # pulse width of delta approximation
pulse_height = 1e15  # pulse height of delta approximation
delta_t = 5  # pulse delay
laser_r = np.array([1, 1])  # illuminated radii of first and second pulse
laser_f = np.array([1, 1])  # frequencies of first and second laser pulse
laser_p = np.array([1, 1])  # time-averaged laser powers of first and second pulse

# time-resolved photocurrent simulation
delta_t_sweep = (-50, 50)  # delta_t range
delta_t_resolution = int(1e1)  # resolution
extractions = np.array([1, 1])  # exciton extraction factors

########################################################################################################################
time_vals = np.linspace(time_range[0], time_range[1], diff_solver_resolution)
delta_t_step = int(delta_t / ((time_range[1] - time_range[0]) / diff_solver_resolution))
N0 = absorption_coeff / (laser_f * np.pi * np.square(laser_r) * pulse_energies)  # initial exciton densities
print(N0)


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


def excitation_function(t, smearing, E):
    """
    Model excitation pulse as delta function for both monolayers using heaviside function as energy response mechanism.

    :param t: time
    :type t: float
    :param smearing: pulsewidth of delta approximation
    :type smearing: float
    :return: excitation factors for both monolayers
    :rtype: 2d-array
    """
    g1 = N0[0] * np.heaviside(E - energy_thresholds[0], 1) * delta_approx(t, smearing=smearing)
    g2 = N0[1] * np.heaviside(E - energy_thresholds[1], 1) * delta_approx(t, smearing=smearing)
    return np.array([g1, g2])


def rhs_coupled_photoresponse(t, Nvec, E):
    """
    Right hand side of the non-linear coupled photoresponse model.

    :param t: time
    :type t: float
    :param Nvec: value of exciton function density
    :type Nvec: 2d-array
    :return: left hand side
    :rtype: 2d-array
    """
    Gt = excitation_function(t, smearing=pulse_width, E=E)
    dN1 = Gt[0] - Nvec[0] / tau[0] - gamma[0] * Nvec[0] ** 2 + alpha[0] * Nvec[1]
    dN2 = Gt[1] - Nvec[1] / tau[1] - gamma[1] * Nvec[1] ** 2 + alpha[1] * Nvec[0]
    return np.array([dN1, dN2])


def single_pulse_photoresponse(t_span, t_eval, N_init, E):
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
                         vectorized=True, dense_output=True, args=(E,))
    return solution


def two_pulses_photoresponse(t_eval, delta_t_steps, N_first_pulse, E):
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
    narray = np.zeros((2, len(t_eval)))
    narray[:, :delta_t_steps] = N_first_pulse.sol(t_eval[:delta_t_steps])
    N0_delta_t = N0 + N_first_pulse.sol(t_eval[delta_t_steps])
    nsol = solve_ivp(rhs_coupled_photoresponse, t_span=(t_eval[delta_t_steps], t_eval[-1]),
                     t_eval=np.linspace(t_eval[delta_t_steps], t_eval[-1], diff_solver_resolution),
                     y0=N0_delta_t, method='RK45', vectorized=True, rtol=0.01, dense_output=True, args=(E,))
    narray[:, delta_t_steps:] = nsol.sol(t_eval[delta_t_steps:])

    return nsol, narray


def lock_in_photocurrent_continuous(delta_t, single_pulse, t_span, t_eval, a_fac):
    a, b = t_span
    n2p, n2p_vals = two_pulses_photoresponse(t_eval, delta_t, single_pulse)

    def integrand(t):
        if t < delta_t:
            pass
        else:
            return a_fac[0] * (n2p.sol(t)[0] - single_pulse.sol(t)[0]) + a_fac[1] * (
                    n2p.sol(t)[1] - single_pulse.sol(t)[1])

    pc = quad(integrand, a, b)

    return pc


def lock_in_photocurrent_discrete(delta_t, single_pulse, t_eval, a_fac, E2):
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
    n2p, n2p_vals = two_pulses_photoresponse(t_eval, delta_t, single_pulse, E=E2)
    integrand = a_fac[0] * (n2p_vals[0] - single_pulse.y[0]) + a_fac[1] * (n2p_vals[1] - single_pulse.y[1])

    pc = trapz(integrand)

    return pc


def photocurrent_delta_t_continous(delta_t_range, pc_resolution, single_pulse, t_span, t_eval, a_fac):
    tstep = (t_span[1] - t_span[0]) / len(t_eval)
    trange = np.linspace(int(delta_t_range[0] / tstep) + 1, int(delta_t_range[1] / tstep), pc_resolution, dtype=int)
    pc_vals = np.zeros_like(trange)
    for i, dt in enumerate(trange):
        print(f'\r{i / len(trange) * 100:2.2f} %', end='')
        pc_vals[i] = lock_in_photocurrent_continuous(dt, single_pulse, t_span, t_eval, a_fac)[0]

    return trange * tstep, pc_vals


def photocurrent_delta_t_discrete(delta_t_range, pc_resolution, single_pulse, t_span, t_eval, a_fac, E2):
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
    tstep = (t_span[1] - t_span[0]) / len(t_eval)
    if abs(delta_t_range[0] / tstep) < 0.5:
        start = 1
    else:
        start = int(delta_t_range[0] / tstep)
    trange = np.linspace(start, int(delta_t_range[1] / tstep), pc_resolution, dtype=int)
    pc_vals = np.zeros_like(trange)
    for i, dt in enumerate(trange):
        print(f'\r{i / len(trange) * 100:2.2f} %', end='')
        pc_vals[i] = lock_in_photocurrent_discrete(dt, single_pulse, t_eval, a_fac, E2)

    return trange * tstep, pc_vals


single_pulse = single_pulse_photoresponse(t_span=time_range, t_eval=time_vals, N_init=N0, E=pulse_energies[0])
double_pulse, double_pulse_vals = two_pulses_photoresponse(t_eval=time_vals, delta_t_steps=delta_t_step,
                                                           N_first_pulse=single_pulse, E=pulse_energies[1])

pc_sim = photocurrent_delta_t_discrete(delta_t_sweep, pc_resolution=delta_t_resolution, single_pulse=single_pulse,
                                       t_span=time_range, t_eval=time_vals, a_fac=extractions, E2=pulse_energies[1])

first_excitation = np.array([excitation_function(t, smearing=pulse_width, E=pulse_energies[0]) for t in time_vals])
second_excitation = np.array(
    [excitation_function(t - time_vals[delta_t_step], smearing=pulse_width, E=pulse_energies[1]) for t in time_vals])

float_formatter = '{:.4f}'.format
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
          f'$E_t$={energy_thresholds} \n' \
          f'$E_p$={pulse_energies}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.65, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

fig1, ax1 = plt.subplots()
ax1.plot(double_pulse.t - delta_t, double_pulse_vals[0], '-')
ax1.plot(double_pulse.t - delta_t, double_pulse_vals[1], '-')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$N(t)$')
ax1.set_title(f'resolution={diff_solver_resolution}')
textstr = f'$\\alpha$={alpha} \n' \
          f'$\\tau$={tau} \n' \
          f'$\\gamma$={gamma} \n' \
          f'$N_0$={N0} \n' \
          f'$E_t$={energy_thresholds} \n' \
          f'$E_p$={pulse_energies} \n' \
          f'$a$={extractions}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.6, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
ax1.set_xlim(-1, 30)
plt.show()
