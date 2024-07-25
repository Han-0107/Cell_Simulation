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
T_swi = 1 @ u_ns

# 定义电源电压
circuit.V(1, 'VDD', circuit.gnd, V_dd)
circuit.V(2, 'VSS', circuit.gnd, 0 @u_V)

# 定义输入信号
# circuit.PulseVoltageSource('Vpulse', 'a', circuit.gnd, initial_value=0 @u_V, pulsed_value=V_dd, 
#                            delay_time=0 @u_ns, rise_time=1 @u_ns, fall_time=1 @u_ns, pulse_width=20 @u_ns, period=40 @u_ns)
# circuit.VoltageSource(3, 'b', circuit.gnd, V_dd)
V_1_up = 0.1*V_dd
V_3_up = 0.3*V_dd
V_5_up = 0.5*V_dd
V_7_up = 0.7*V_dd
V_9_up = 0.9*V_dd
V_1_down = 0.9*V_dd
V_3_down = 0.7*V_dd
V_5_down = 0.5*V_dd
V_7_down = 0.3*V_dd
V_9_down = 0.1*V_dd
T_pulse = 20 @ u_ns
T_period = 40 @ u_ns

circuit.PieceWiseLinearVoltageSource('Vpulse', 'a', circuit.gnd,
                                        values=[
                                                (0, 0), 
                                                (0.1*T_swi, V_1_up),
                                                (0.3*T_swi, V_3_up),
                                                (0.5*T_swi, V_5_up),
                                                (0.7*T_swi, V_7_up),
                                                (0.9*T_swi, V_9_up),
                                                (T_swi, V_dd), 
                                                (T_swi+T_pulse, V_dd), 
                                                (1.1*T_swi+T_pulse, V_1_down),
                                                (1.3*T_swi+T_pulse, V_3_down),
                                                (1.5*T_swi+T_pulse, V_5_down),
                                                (1.7*T_swi+T_pulse, V_7_down),
                                                (1.9*T_swi+T_pulse, V_9_down),                                                
                                                (2*T_swi+T_pulse, 0), 
                                                (2*T_swi+T_period, 0)
                                                ]
                                    )

# # 电阻和电容元件（可选，视实际情况）
# circuit.R('inA', 'a', 'inA', R_in)
# circuit.R('inB', 'b', 'inB', R_in)
# circuit.R('out', 'y', 'out', R_out)
# circuit.C('inA', 'inA', circuit.gnd, C_in)
# circuit.C('inB', 'inB', circuit.gnd, C_in)
# circuit.C('out', 'out', circuit.gnd, C_out)

# 定义门
circuit.X(1, 'INVX1', 'y', 'a', 'VDD', 'VSS')

# 进行瞬态仿真
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=0.001 @u_ns, end_time=45 @u_ns)

# 计算延时
def calculate_delay(time, signal, threshold):
    rise_times = time[np.where(np.diff(signal > threshold, prepend=False))]
    fall_times = time[np.where(np.diff(signal < threshold, prepend=False))]
    if len(rise_times) > 1:
        tpLH = rise_times[1] - rise_times[0]
    else:
        tpLH = None
    if len(fall_times) > 1:
        tpHL = fall_times[1] - fall_times[0]
    else:
        tpHL = None
    return tpLH, tpHL

tpLH, tpHL = calculate_delay(np.array(analysis.time), np.array(analysis['y']), float(V_dd)/2)

print(tpLH, "ns")
print(tpHL, "ns")

figure, ax = plt.subplots(figsize=(10, 6))
plot(analysis['a'], axis=ax, label='V(in1)')
plot(analysis['y'], axis=ax, label='V(out)')
plt.title('Gate Transient Analysis')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()
