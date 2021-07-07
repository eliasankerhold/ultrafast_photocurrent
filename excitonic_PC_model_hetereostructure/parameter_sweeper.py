import subprocess
import numpy as np
import itertools

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


# CONSTANT PARAMETERS
# exciton density simulation
time_range = (0, 101e-11)  # time range
diff_solver_resolution = int(1e6)  # resolution
# time-resolved photocurrent simulation
delta_t_sweep = (-10e-11, 5e-12)  # delta_t range
delta_t_resolution = int(5e1 + 1)  # resolution

# SWEEPABLE PARAMETERS
# device - first entry is MoSe2 (top), second is MoS2 (bottom)
alpha_mos2 = smart_range_creator(1, 2, 10)
alpha_mose2 = smart_range_creator(1)
gamma_mos2 = smart_range_creator(1)
gamma_mose2 = smart_range_creator(1)
tau_mos2 = smart_range_creator(1)
tau_mose2 = smart_range_creator(1)
# setup
pulse_one_energy = smart_range_creator(1, 5, 2)
pulse_two_energy = smart_range_creator(1)
power_one = smart_range_creator(1)
power_two = smart_range_creator(1)

total_variations = len(alpha_mos2) * len(alpha_mose2) * len(gamma_mos2) * len(gamma_mose2) * len(tau_mos2) * \
                   len(tau_mose2) * len(pulse_one_energy) * len(pulse_two_energy) * len(power_one) * len(power_two)

packed_list = [alpha_mos2, alpha_mose2, gamma_mos2, gamma_mose2, tau_mos2, tau_mose2, pulse_one_energy,
               pulse_two_energy, power_one, power_two]

combinations = itertools.combinations(packed_list, len(packed_list))
for i in combinations:
    print(i)

number_of_tasks = 2
save_index = 1
folder_name = 'test'

params = parameter_packer(number_of_tasks, [alpha_mos2, alpha_mose2], [gamma_mos2, gamma_mose2], [tau_mos2, tau_mose2],
                          [pulse_one_energy, pulse_two_energy], [power_one, power_two], time_range,
                          diff_solver_resolution, delta_t_sweep, delta_t_resolution, save_index, folder_name)
# print(params)
# p1 = subprocess.Popen(f'multi_core_simulation.py {params}', stdout=subprocess.PIPE, shell=True)
# p2 = subprocess.Popen('multi_core_simulation.py', stdout=subprocess.PIPE, shell=True)
# p3 = subprocess.Popen('multi_core_simulation.py', stdout=subprocess.PIPE, shell=True)
# out, err = p1.communicate()
# print(out)
