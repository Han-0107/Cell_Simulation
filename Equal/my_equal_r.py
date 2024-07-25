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

V_dd = 2.1 @u_V
R_in = 1 @u_Ohm
C_in = 1 @u_pF
R_out = 1 @u_Ohm
C_out = 1 @u_pF

# 定义电源电压
circuit.V(1, 'VDD', circuit.gnd, V_dd)
circuit.V(2, 'VSS', circuit.gnd, 0 @u_V)

# 定义输入信号
# circuit.PulseVoltageSource('Vpulse', 'a', circuit.gnd, initial_value=0 @u_V, pulsed_value=V_dd, 
#                            delay_time=0 @u_ns, rise_time=1 @u_ns, fall_time=1 @u_ns, pulse_width=20 @u_ns, period=40 @u_ns)
circuit.V('Vin', 'a', circuit.gnd, V_dd)
# 定义门
circuit.X(1, 'INVX1', 'y', 'a', 'VDD', 'VSS')

# 进行直流工作点分析
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
dc_analysis = simulator.operating_point()

print("Nodes:")
for node in dc_analysis.nodes:
    print(f'{node}: {float(dc_analysis[node]):.4f} V')

print("\nBranches:")
for branch in dc_analysis.branches:
    print(f'{branch}: {float(dc_analysis[branch]):.10f} A')

# 获取电压和电流
v_out = float(dc_analysis['y'])
v_in = float(dc_analysis['a'])
i_in = float(dc_analysis['vvin'])

# 计算等效电阻
r_eq = (v_in - v_out) / i_in

print(f'等效电阻为: {r_eq} Ohm')
