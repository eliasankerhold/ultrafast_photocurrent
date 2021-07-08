import subprocess
import numpy as np
from multiprocessing import cpu_count
import sys

# # NATURAL CONSTANTS
h = 6.62607015e-34
c = 2.99792458e8
e = 1.602176634e-19


def parameter_packer(tasks, alphas, gammas, taus, energies, powers, time_range, time_resolution, delta_t_range,
                     delta_t_resolution, save_index, folder_name):
    packed_string = f'{tasks} {alphas[0]} {alphas[1]} {gammas[0]} {gammas[1]} {taus[0]} {taus[1]} {energies[0]} ' \
                    f'{energies[1]} {powers[0]} {powers[1]} {time_range[0]} {time_range[1]} {time_resolution} ' \
                    f'{delta_t_range[0]} {delta_t_range[1]} {delta_t_resolution} {save_index} {folder_name}'

    return packed_string


def smart_range_creator(start, stop=None, resolution=None, scale='lin'):
    if stop is None:
        return np.array([start], dtype=float)
    elif scale == 'lin':
        return np.linspace(start, stop, resolution, dtype=float)
    elif scale == 'log':
        return np.logspace(start, stop, resolution, dtype=float)


def permutations_creator(length1, length2, *args):
    """
    Creates a list of all possible index permutations out of input lengths.

    :param length1: length of first array
    :type length1: int
    :param length2: length of second array
    :type length2: int
    :param args: lengths of more arrays
    :type args: int
    :return: array of permutations, number of permutations
    :rtype: nd-array, int
    """
    lengths = [length1, length2, *args]
    total = np.prod(lengths)
    tot = total
    perms = np.zeros((total, len(lengths)), dtype=int)
    for i, l in enumerate(lengths):
        indexspace = np.arange(0, l, 1, dtype=int)
        block_one = np.repeat(indexspace, int(tot / l))
        perms[:, i] = np.tile(block_one, total // len(block_one))
        tot /= l

    return perms, total


# CONSTANT PARAMETERS
# exciton density simulation
time_range = (0, 101e-11)  # time range
diff_solver_resolution = int(1e6)  # resolution
# time-resolved photocurrent simulation
delta_t_sweep = (-10e-11, 5e-12)  # delta_t range
delta_t_resolution = int(5e1 + 1)  # resolution

# SWEEPABLE PARAMETERS
# device - first entry is MoSe2 (top), second is MoS2 (bottom)
alpha_mos2 = smart_range_creator(1, 2, 2)
alpha_mose2 = smart_range_creator(1)
gamma_mos2 = smart_range_creator(1)
gamma_mose2 = smart_range_creator(1, 6, 3)
tau_mos2 = smart_range_creator(1)
tau_mose2 = smart_range_creator(1)
# setup
pulse_one_energy = smart_range_creator(1, 2, 1) * e
pulse_two_energy = smart_range_creator(1) * e
power_one = smart_range_creator(1)
power_two = smart_range_creator(1)

permutations, total_perms = permutations_creator(len(alpha_mos2), len(alpha_mose2), len(gamma_mos2), len(gamma_mose2),
                                                 len(tau_mos2), len(tau_mose2), len(pulse_one_energy),
                                                 len(pulse_two_energy), len(power_one), len(power_two))

packed_list = [alpha_mos2, alpha_mose2, gamma_mos2, gamma_mose2, tau_mos2, tau_mose2, pulse_one_energy,
               pulse_two_energy, power_one, power_two]

logical_cores = cpu_count()
number_of_tasks = 3
number_of_subprocesses = 4
save_index = 1
folder_name = 'test'

if number_of_tasks * number_of_subprocesses > logical_cores:
    print(f'TOO MANY TASKS! {logical_cores} logical cores were found. Cannot run {number_of_subprocesses} processes with '
          f'{number_of_tasks} tasks each.')
    sys.exit('Aborted for precaution.')

param_collection = []
sps = []

# os.system("cmd /k echo I am working on it!")

for i, p in enumerate(permutations):
    param_collection.append(
        parameter_packer(number_of_tasks, [alpha_mos2[p[0]], alpha_mose2[p[1]]], [gamma_mos2[p[2]], gamma_mose2[p[3]]],
                         [tau_mos2[p[4]], tau_mose2[p[5]]], [pulse_one_energy[p[6]], pulse_two_energy[p[7]]],
                         [power_one[p[8]], power_two[p[9]]], time_range, diff_solver_resolution, delta_t_sweep,
                         delta_t_resolution, i, folder_name))

param_collection = [param_collection[i:i + number_of_subprocesses] for i in
                    range(0, len(param_collection), number_of_subprocesses)]

for i, paramset in enumerate(param_collection):
    for k, sp in enumerate(range(number_of_subprocesses)):
        sps.append(subprocess.Popen(f'multi_core_simulation.py {paramset[k]}', stdout=subprocess.PIPE, shell=True))
    for sp in sps:
        sp.communicate()
        sp.terminate()
    sps = []
