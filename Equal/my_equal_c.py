#!/usr/bin/python3
import matplotlib.pyplot as plt
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

# 定义电源电压
circuit.V(1, 'VDD', circuit.gnd, V_dd)
circuit.V(2, 'VSS', circuit.gnd, 0 @u_V)

# 定义输入信号
# circuit.PulseVoltageSource('Vpulse', 'a', circuit.gnd, initial_value=0 @u_V, pulsed_value=V_dd,
#                            delay_time=0 @u_ns, rise_time=1 @u_ns, fall_time=1 @u_ns, pulse_width=20 @u_ns, period=40 @u_ns)
circuit.V('Vin', 'a', circuit.gnd, V_dd)
# 定义 NAND 门
circuit.X(1, 'INVX1', 'y', 'a', 'VDD', 'VSS')

# 进行小信号交流仿真
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
ac_analysis = simulator.ac(start_frequency=1@u_Hz, stop_frequency=1@u_GHz, number_of_points=1000, variation='dec')

# 提取并处理仿真结果
frequency = ac_analysis.frequency.as_ndarray()
v_in = np.abs(ac_analysis['a'].as_ndarray())  # 输入电压幅值
i_in = np.abs(ac_analysis['vvin'].as_ndarray())  # 输入电流幅值

# 计算输入电容
c_in = i_in / (2 * np.pi * frequency * v_in)
print(frequency)
print(v_in)
print(c_in)
