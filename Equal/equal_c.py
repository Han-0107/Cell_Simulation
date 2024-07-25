import matplotlib.pyplot as plt
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import *
from PySpice.Unit import *
import numpy as np

# 创建电路
circuit = Circuit('INV')

# 包含 SPICE 模型库
circuit.include('/home/yaohui/Research/PySpice/Libs/cells.sp')
circuit.include('/home/yaohui/Research/PySpice/Libs/gpdk45nm.m')

V_dd  = 2.1 @u_V
R_in  = 1 @u_Ohm
C_in  = 1 @u_pF
R_out = 1 @u_Ohm
C_out = 1 @u_pF

# 定义电源电压
circuit.V(1, 'VDD', circuit.gnd, V_dd)
circuit.V(2, 'VSS', circuit.gnd, 0 @u_V)

# 定义输入信号
circuit.PulseVoltageSource('Vpulse', 'a', circuit.gnd, initial_value=0 @u_V, pulsed_value=V_dd, 
                           delay_time=0 @u_ns, rise_time=1 @u_ns, fall_time=1 @u_ns, pulse_width=20 @u_ns, period=40 @u_ns)

# 定义 NAND 门
circuit.X(1, 'INVX1', 'y', 'a', 'VDD', 'VSS')

# 进行小信号交流仿真
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
ac_analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, number_of_points=1000, variation='dec')

print("Nodes:")
for node in ac_analysis.nodes:
    print(f'{node}: {float(ac_analysis[node]):.4f} V')

print("\nBranches:")
for branch in ac_analysis.branches:
    print(f'{branch}: {float(ac_analysis[branch]):.10f} A')

# # 提取输入电容
# v_in = np.abs(np.array([ac_analysis['a'].as_ndarray()]))
# i_in = np.abs(np.array([ac_analysis['Vpulse'].as_ndarray()]))

# c_in = i_in / (2 * np.pi * ac_analysis.frequency.as_ndarray() * v_in)

# # 绘制等效电容曲线
# plt.figure()
# plt.semilogx(ac_analysis.frequency, c_in[0], label='C_in')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Capacitance [F]')
# plt.title('Equivalent Capacitance vs Frequency')
# plt.legend()
# plt.grid(True)
# plt.show()
