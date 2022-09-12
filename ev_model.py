# This notebook plots a torque-speed curve for a DC motor with a lossless DC/DC converter and a battery with an internal resistance.

# TODO: add logic to tell you which discriminant died how
# TODO: add gearing to vehicle torque speed and load curve
# TODO: break out power graph and load modeling graph and predicted/reality model
# TODO: come up with bicycle test data collection runs (100 phase / 75 battery?)
# TODO: plot motor efficiency
# TODO: create function using power equality equation with peak powers on motor side and plot and look for differences and better match with experiment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pint import UnitRegistry
# u = UnitRegistry()
import sympy
sympy.init_printing()

def define_variables():
    Ib = sympy.symbols('I_b', positive=True)           # battery current during motoring
    Ic = sympy.symbols('I_c', positive=True)           # motor current
    Vb = sympy.symbols('V_b', positive=True)           # battery voltage
    Vc = sympy.symbols('V_c', positive=True)           # motor voltage line to line (should be line phase?)
    Rm = sympy.symbols('R_m', positive=True)           # single phase motor resistance
    k = sympy.symbols('k', positive=True)              # line to line voltage constant
    omega = sympy.symbols('omega', positive=True)      # rotor angular velocity
    Rb = sympy.symbols('R_b', positive=True)           # battery internal resistance
    Voc = sympy.symbols('V_oc', positive=True)         # battery open circuit voltage
    Imm = sympy.symbols('I_mm', positive=True)         # controller maximum current
    Ibm = sympy.symbols('I_bm', positive=True)         # battery maximum current
    torque = sympy.symbols('tau')                      # motor torque

    return Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque

def define_equations_dc(Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):

    power_eq = sympy.Eq(Vb * Ib, Vc * Ic)                           # controller energy balance
    controller_loop = sympy.Eq(Vc, Ic * Rm + k * omega)              # controller side voltage loop
    battery_loop = sympy.Eq(Voc, Ib * Rb + Vb)                       # battery side voltage loop
    regime_1_eq = sympy.Eq(Ic, Imm)
    regime_2_eq = sympy.Eq(Ib, Ibm)
    regime_3_eq = sympy.Eq(Vb , Vc)
    return power_eq, controller_loop, battery_loop, regime_1_eq, regime_2_eq, regime_3_eq

def define_equations_3p(Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):

    power_eq = sympy.Eq(Vb * Ib, Vc * Ic)                    # controller energy balance
    power_eq = sympy.Eq(Vb * Ib, Vc * Ic * 3 / 2)                    # controller energy balance
    # reducing k by sqrt(2) makes a good fit but I'm expecting sqrt(3) from line to line models
    controller_loop = sympy.Eq(Vc, Ic * Rm + k * omega)              # controller side voltage loop
    controller_loop = sympy.Eq(Vc, Ic * Rm + k * omega / sympy.sqrt(3))              # controller side voltage loop
    battery_loop = sympy.Eq(Voc, Ib * Rb + Vb)                       # battery side voltage loop
    regime_1_eq = sympy.Eq(Ic, Imm)
    regime_2_eq = sympy.Eq(Ib, Ibm)
    regime_3_eq = sympy.Eq(Vb, Vc)
    regime_3_eq = sympy.Eq(Vb / sympy.sqrt(3), Vc)
    return power_eq, controller_loop, battery_loop, regime_1_eq, regime_2_eq, regime_3_eq

def solve_equations(power_eq, controller_loop, battery_loop, regime_1_eq, regime_2_eq, regime_3_eq,
                    Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):
    # solve for phase current limited regime
    phase_limit = sympy.solve([power_eq, battery_loop, controller_loop, regime_1_eq], exclude=[Rb, Rm, k, Imm, Voc])[0]
    battery_limit = sympy.solve([power_eq, battery_loop, controller_loop, regime_2_eq], exclude=[Rb, Rm, k, Ibm, Voc])[0]
    duty_limit = sympy.solve([power_eq, battery_loop, controller_loop, regime_3_eq], exclude=[Rb, Rm, k, Voc])[0]
    return phase_limit, battery_limit, duty_limit

def display_equations(phase_limit, battery_limit, duty_limit):
    from IPython.display import display
    display(phase_limit)
    display(battery_limit)
    display(duty_limit)

def calculate_crossover_frequencies(phase_limit, battery_limit, duty_limit, ev_system,
                                     Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):
    crossover_1_2 = sympy.solve(sympy.Eq(Ibm, phase_limit[Ib]), omega)[0]
    omega_1_2 = crossover_1_2.evalf(subs=ev_system)
    crossover_2_3 = sympy.solve(sympy.Eq(battery_limit[Ic], duty_limit[Ic]), omega)[0]
    omega_2_3 = crossover_2_3.evalf(subs=ev_system)
    crossover_3 = sympy.solve(sympy.Eq(duty_limit[Ic], 0), omega)[0]
    omega_3 = crossover_3.evalf(subs=ev_system)
    assert 0 < omega_1_2 < omega_2_3 < omega_3
    return omega_1_2, omega_2_3, omega_3

def generate_currents(ev_system, phase_limit, battery_limit, duty_limit, Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):
    omega_1_2, omega_2_3, omega_3 = calculate_crossover_frequencies(phase_limit, battery_limit, duty_limit, ev_system,
                                                                   Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque)
    regime_1_omega = np.linspace(0, float(omega_1_2), 100, endpoint=False)
    regime_2_omega = np.linspace(float(omega_1_2), float(omega_2_3), 100, endpoint=False)
    regime_3_omega = np.linspace(float(omega_2_3), float(omega_3), 100)

    Ib_1 = list(map(sympy.lambdify(omega, phase_limit[Ib].subs(ev_system)), regime_1_omega))     # phase limited battery current
    Ic_1 = list(map(sympy.lambdify(omega, phase_limit[Ic].subs(ev_system)), regime_1_omega))
    Ib_2 = list(map(sympy.lambdify(omega, battery_limit[Ib].subs(ev_system)), regime_2_omega))
    Ic_2 = list(map(sympy.lambdify(omega, battery_limit[Ic].subs(ev_system)), regime_2_omega))   # battery limited phase current
    Ib_3 = list(map(sympy.lambdify(omega, duty_limit[Ib].subs(ev_system)), regime_3_omega))
    Ic_3 = list(map(sympy.lambdify(omega, duty_limit[Ic].subs(ev_system)), regime_3_omega))
    Ib_values = np.hstack((np.array(Ib_1), np.array(Ib_2), np.array(Ib_3)))
    Ic_values = np.hstack((np.array(Ic_1), np.array(Ic_2), np.array(Ic_3)))
    omega_values = np.hstack((regime_1_omega, regime_2_omega, regime_3_omega))

    return omega_values, Ib_values, Ic_values

def generate_voltages(ev_system, phase_limit, battery_limit, duty_limit, Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque):
    omega_1_2, omega_2_3, omega_3 = calculate_crossover_frequencies(phase_limit, battery_limit, duty_limit, ev_system, Ib, Ic, Vb, Vc, Rm, k, omega, Rb, Voc, Imm, Ibm, torque)

    regime_1_omega = np.linspace(0, float(omega_1_2), 100, endpoint=False)
    regime_2_omega = np.linspace(float(omega_1_2), float(omega_2_3), 100, endpoint=False)
    regime_3_omega = np.linspace(float(omega_2_3), float(omega_3), 100)

    Vb_1 = list(map(sympy.lambdify(omega, phase_limit[Vb].subs(ev_system)), regime_1_omega))   # phase limited battery current
    Vc_1 = list(map(sympy.lambdify(omega, phase_limit[Vc].subs(ev_system)), regime_1_omega))
    Vb_2 = list(map(sympy.lambdify(omega, battery_limit[Vb].subs(ev_system)), regime_2_omega))
    Vc_2 = list(map(sympy.lambdify(omega, battery_limit[Vc].subs(ev_system)), regime_2_omega))   # battery limited phase current
    Vc_3 = list(map(sympy.lambdify(omega, duty_limit[Vc].subs(ev_system)), regime_3_omega))
    Vb_3 = list(map(sympy.lambdify(omega, duty_limit[Vb].subs(ev_system)), regime_3_omega))

    Vb_values = np.hstack((np.array(Vb_1), np.array(Vb_2), np.array(Vb_3)))
    Vc_values = np.hstack((np.array(Vc_1), np.array(Vc_2), np.array(Vc_3)))
    omega_values = np.hstack((regime_1_omega, regime_2_omega, regime_3_omega))

    return omega_values, Vb_values, Vc_values

def read_dashboard_ride_data(filename):
    return pd.read_csv(filename, sep=';')

def plot_dashboard_current_speed(data, ax):
    data['rad_per_sec'] = data['erpm'] / 23 / 60 * 2 * np.pi
    # fig, ax = plt.subplots()
    ax.plot(data['rad_per_sec'], data['current_motor'], label='observed motor current')

def plot_motor_current(omega, motor_current, ax):
    ax.plot(omega.magnitude, motor_current.magnitude, label='predicted motor current')

def plot_powers(omega, motor_heat, mechanical_power, battery_power, load_power, ax):
    omega = omega.magnitude
    ax.plot(omega, motor_heat.magnitude, label='Motor Heat')
    ax.plot(omega, mechanical_power.magnitude, label='Mechanical Power')
    ax.plot(omega, battery_power.magnitude, label='Electrical Power')
    ax.plot(omega, load_power.magnitude, label='Load Power')
    ax.legend()
    ax.grid()
    plt.show()

def get_parameters():
    # return as block?
    pass

# Voc = 58 * u.volt
# Rb = 0.141 * u.ohm
# Immax = 150 * u.amp
# Ibmax = 100 * u.amp
# Rm = 0.06 * u.ohm
# k = 1.2 * u.volt * u.sec / u.rad
# omega_no_load = Voc / k

def calculate_load_torque_geared(omega_wheel, ev_parameters, u, gear_ratio=None):

    torque_wheel = (1 / 2 *
                    ev_parameters['density'] *
                    ev_parameters['r']**3 *
                    ev_parameters['drag_coefficient'] *
                    ev_parameters['area'] *
                    omega_wheel**2).to(u.newton * u.meter)
    # when gear_ratio > 1, torque_wheel > torque_motor
    torque_motor = torque_wheel * gear_ratio

    return torque_motor

def plot_gear_torques(high_gear_ratio, low_gear_ratio, ev_parameters, u, num_gears=4):

    k = (1 / ev_parameters['Kv']).to(u.volt * u.sec / u.rad)
    omega_motor_max = ev_parameters['Voc'] / k
    omega_motor = np.linspace(0, omega_motor_max, 51)

    # evenly spaced gear_ratio ratios
    gear_ratios = np.logspace(np.log10(high_gear_ratio), np.log10(low_gear_ratio), num_gears)
    torques = {}
    fig, ax = plt.subplots()
    for gear_ratio in gear_ratios:
        omega_wheel = omega_motor * gear_ratio # slower wheel for lower gear ratio
        speed_mps = omega_wheel * ev_parameters['r']
        torques[gear_ratio] = calculate_load_torque_geared(omega_wheel, ev_parameters, u, gear_ratio=gear_ratio)
        ax.plot(omega_wheel.magnitude, torques[gear_ratio].magnitude, label=f'ratio = {gear_ratio:.3f}')

    speed_ax = ax.secondary_xaxis(-0.2, functions=(lambda x: x * ev_parameters['r'].magnitude,
                                                   lambda x: x / ev_parameters['r'].magnitude))
    speed_ax.set_xlabel('Vehicle Speed (m/sec)')
    current_ax = ax.secondary_yaxis('right', functions=(lambda x: x / k.magnitude,
                                                        lambda x: x * k.magnitude))
    current_ax.set_ylabel('Motor Current (A)')
    ax.set_xlabel('Wheel angular speed (rad/sec)')
    ax.set_ylabel('Rotor Torque (Nm)')
    ax.set_title('Rotor Torque and Gear Ratios')
    ax.legend()
    plt.show()


def calculate_currents_voltages(Voc, Rb, Rm, k, Immax, Ibmax, u):
    # returns battery current, motor current, and battery voltage
    # TODO: consider cleaner decomposition
    # TODO: how to deal with multiply defined registries

    omega_no_load = Voc / k
    omega = np.linspace(0, omega_no_load, 101)
    # Regime 1 (Ic at Icmax) battery current
    Ib = (Voc / Rb - np.sqrt(Voc**2 / Rb**2 - 4 * (Immax**2 * Rm + Immax * k * omega) / Rb)) / 2
    Ib = (Voc - np.sqrt(-6*Immax**2 * Rb * Rm - 6 * Immax * Rb * k*omega + Voc**2))/(2*Rb)
    # Regime 2 (Ib at Ibmax) motor current
    Ic = (- k * omega / Rm + np.sqrt(k**2 * omega**2 / Rm**2 + 4 * Ibmax * (Voc - Ibmax * Rb) / Rm)) / 2
    Ic = -(k*omega/2 - np.sqrt(-24*Ibmax**2*Rb*Rm + 24*Ibmax*Rm*Voc + 9*k**2*omega**2)/6)/Rm
    # Regime 3 (Ib equal to Ic) current
    I = (Voc - k * omega) / (Rm + Rb)
    I = (Voc + np.sqrt(3)*k*omega)/(Rb + 2*Rm)

    # Regime 1 (Ic at Icmax) battery voltage
    Vb1 = (Voc + np.sqrt(Voc**2 - 4 * Immax**2 * Rb * Rm - 4 * Immax * Rb * k * omega)) / 2
    # replace NaN so that comparisons work
    Vb1 = np.nan_to_num(Vb1.magnitude, nan=0) * u.volt
    Vb2 = (Voc - Ibmax * Rb) * np.ones(len(omega))
    Vb3 = (Voc - I * Rb)

    # concatenate battery voltages
    Vbfull = np.concatenate((Vb1[Vb1 > Vb2],
                            Vb2[(Vb2 > Vb1) & (Vb2 > Vb3)],
                            Vb3[Vb3 > Vb2]))

    # concatenate battery currents
    Ib1 = Ib[Ib < Ibmax]
    Ib3 = I[I < Ibmax]
    Ib2 = Ibmax * np.ones(len(omega) - len(Ib1) - len(Ib3))
    Ibfull = np.concatenate((Ib1, Ib2, Ib3))

    # concatenate motor currents
    Ic2 = Ic[(Ic < Immax) &  (Ic < I)]
    Ic3 = I[(I < Immax) & (I < Ic)]
    Ic1 = Immax * np.ones(len(omega) - len(Ic2) - len(Ic3))
    Icfull = np.concatenate((Ic1, Ic2, Ic3))

    plt.plot(Ic1)
    plt.plot(Ic2)
    plt.plot(Ic3)

    return omega, Ibfull, Icfull, Vbfull

def calculate_load_torque(omega, u):

    density = 1.2 * u.kg / u.meter**3
    r = 0.64 * u.meter / u.rad / 2
    drag_coefficient = 1.0
    area = 0.504 * u.meter**2

    load_torque = 1 / 2 * density * r**3 * drag_coefficient * area * omega**2

    return load_torque

def plot_currents_and_voltages(omega, Ib, Ic, Vb):
    # TODO: what is up with warning about sqrt invalid value?
    fig, ax = plt.subplots()
    plt.plot(omega.magnitude, Ib.magnitude, label='Battery Current')
    plt.plot(omega.magnitude, Ic.magnitude, label='Motor Current')
    plt.plot(omega.magnitude, Vb.magnitude, label='Battery Voltage')
    plt.grid()
    plt.legend()
    ax.set_xlabel('Wheel Angular Speed (rad/sec)')
    plt.show()


def plot_motor_and_load_torque(omega, motor_torque, load_torque):
    # TODO: fix up to use units properly
    def rps2mps(x):
        return x * 60 * r.magnitude * 60 / 1000
    def mps2rps(x):
        return x

    plt.plot(omega, motor_torque, label='Motor')
    plt.plot(omega, load_torque, label='Vehicle Load')
    plt.xlabel('Motor Speed (rad/sec)')
    plt.ylabel('Torque (Nm)')
    plt.grid()
    plt.legend()
    ax = plt.gca()
    secax = ax.secondary_xaxis(-0.2, functions=(rps2mps, mps2rps))
    secax.set_xlabel('Vehicle Speed (kph)')
    fig = plt.gcf()
    ax.set_title('Torque-Speed Curves for Motor and Load')
    fig.savefig('torque-speed.svg', bbox_inches='tight')

def calculate_derived(Ib, Ic, Vb, Rm, k, omega, load_torque):
    motor_torque = Ic * k
    motor_heat = Ic ** 2 * Rm
    mechanical_power = Ic * k * omega
    battery_power = Ib * Vb
    load_power = load_torque * omega
    return motor_torque, motor_heat, mechanical_power, battery_power, load_power

def plot_verbose():
    #plt.plot(omega, Ib)
    #plt.plot(omega, Ibmax * np.ones(len(omega)))
    #plt.plot(omega, I)

    #plt.plot(omega, I)
    #plt.plot(omega, Ic)
    #plt.plot(omega, Immax * np.ones(len(omega)))

    plt.plot(omega, Vb1)#; plt.show()
    plt.plot(omega, Vb2)#; plt.show()
    plt.plot(omega, Vb3)#; plt.show()
    plt.show()
    plt.plot(Vbfull)

def plot_torques(omega, motor_torque, load_torque):

    omega = omega.magnitude
    plt.plot(omega, motor_torque.magnitude, label='Motor Torque')
    plt.plot(omega, load_torque.magnitude, label='Load Torque')
    plt.legend()
    plt.grid()
    plt.show()

def plot_powers_and_torques(omega,
                motor_torque,
                motor_heat,
                mechanical_power,
                battery_power,
                load_power,
                load_torque):

    omega = omega.magnitude
    plt.plot(omega, motor_torque.magnitude, label='Motor Torque')
    plt.plot(omega, motor_heat.magnitude, label='Motor Heat')
    plt.plot(omega, mechanical_power.magnitude, label='Mechanical Power')
    plt.plot(omega, battery_power.magnitude, label='Electrical Power')
    plt.plot(omega, load_power.magnitude, label='Load Power')
    plt.plot(omega, load_torque.magnitude, label='Load Torque')
    plt.legend()
    plt.grid()
    plt.show()
