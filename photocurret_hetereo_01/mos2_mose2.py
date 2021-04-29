import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from matplotlib import pyplot as plt
import numpy as np
from PySpice.Spice.Netlist import DeviceModel

logger = Logging.setup_logging()

ml_mos2 = Circuit('Monolayer MoS2')

print(ml_mos2.model_names)

# test = DeviceModel('test', 'SW')
# print(test)

# circuit parameters
junction_resistance = 25@u_kΩ
junction_capacitance = 10@u_uF
parasitic_capacitance = 0.001@u_pF
tmdc_resistance = 5@u_MΩ

# setup parameters
external_resistance = 10@u_MΩ
chopper_frequency = 50

# simulation parameters
# steps per period
resolution = 20
# number of periods
length = 5
# ambient temp
t = 20


def chopper_frequency_sweep(start, stop, count, current_source, my_simulator, node='out', log=False):
    if not log:
        pw_sweep = np.linspace(1/(4*start), 1/(4*stop), count)
    else:
        pw_sweep = np.logspace(1/(4*start), 1/(4*stop), count)

    res = []

    for pw in pw_sweep:
        current_source.pulse_width = pw@u_s
        current_source.period = 2*pw@u_s
        result = my_simulator.transient(step_time=current_source.period/resolution,
                                        end_time=current_source.period*length)
        res.append(result)

    return res

r_ext = ml_mos2.R(1, ml_mos2.gnd, 'out', external_resistance)
c_p = ml_mos2.C(1, ml_mos2.gnd, 'out', parasitic_capacitance)
c_j1 = ml_mos2.C(2, 'out', 'ml_contact_1', junction_capacitance)
r_j1 = ml_mos2.R(2, 'out', 'ml_contact_1', junction_resistance)
r_m = ml_mos2.R(3, 'ml_contact_1', 'ml_contact_2', tmdc_resistance)
c_j2 = ml_mos2.C(3, 'ml_contact_2', ml_mos2.gnd, junction_capacitance)
r_j2 = ml_mos2.R(4, 'ml_contact_2', ml_mos2.gnd, junction_resistance)
i_photo = ml_mos2.PulseCurrentSource('_photo', 'ml_contact_1', 'out', initial_value=0, pulsed_value=1@u_uA,
                                     pulse_width=1/(4*chopper_frequency)@u_s, period=1/(2*chopper_frequency)@u_s)

print(ml_mos2.models)

print(ml_mos2)
print(f'pulse frequency: {i_photo.frequency}')

simulator = ml_mos2.simulator(temperature=t, nominal_temperature=t)
analysis = simulator.transient(step_time=i_photo.period/100, end_time=i_photo.period*length)

fig1, ax1 = plt.subplots()
ax1.plot(np.array(analysis.out.abscissa), np.array(analysis.out), 'o-')
# ax1.hlines(16.73e-6, 0, 500)

# sweep_result = chopper_frequency_sweep(50, 50e3, 1000, i_photo, simulator)
#
# print(np.array(analysis.out).max())
#
# max_amps = np.zeros(len(sweep_result))
#
# for ind, s in enumerate(sweep_result):
#     max_amps[ind] = np.array(s.out).max()
#
# print(max_amps)

plt.show()