import matplotlib.pyplot as plt
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import *
from PySpice.Unit import *
import numpy as np

# 创建电路
circuit = Circuit('NAND')
gate = 'NAND2X1'

# 包含 SPICE 模型库
circuit.include('/home/yaohui/Research/Cell_Simulation/Libs/cells.sp')
circuit.include('/home/yaohui/Research/Cell_Simulation/Libs/gpdk45nm.m')

V_dd  = 1.1 @u_V
Cap_load = 0.1 @u_pF
T_swi = 0.06 @ u_ns

# 定义电源电压
circuit.V(1, 'VDD', circuit.gnd, V_dd)
circuit.V(2, 'VSS', circuit.gnd, 0 @u_V)

# 负载端的RC
# circuit.R('out', 'y', 'out', R_out)
circuit.C('out', 'y', circuit.gnd, Cap_load)

# 定义输入信号
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
T_pulse = 5 @ u_ns
T_period = 20 @ u_ns

# circuit.PieceWiseLinearVoltageSource('Vpulse', 'a', circuit.gnd,
#                                         values=[
#                                                 (0, 0), 
#                                                 (0.1*T_swi, V_1_up),
#                                                 (0.3*T_swi, V_3_up),
#                                                 (0.5*T_swi, V_5_up),
#                                                 (0.7*T_swi, V_7_up),
#                                                 (0.9*T_swi, V_9_up),
#                                                 (T_swi, V_dd), 
#                                                 (T_swi+T_pulse, V_dd), 
#                                                 (1.1*T_swi+T_pulse, V_1_down),
#                                                 (1.3*T_swi+T_pulse, V_3_down),
#                                                 (1.5*T_swi+T_pulse, V_5_down),
#                                                 (1.7*T_swi+T_pulse, V_7_down),
#                                                 (1.9*T_swi+T_pulse, V_9_down),                                                
#                                                 (2*T_swi+T_pulse, 0), 
#                                                 (2*T_swi+T_period, 0)
#                                                 ]
#                                     )
circuit.PulseVoltageSource('VIN_AN', 'a', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=V_dd,
                           delay_time=0@u_ns, rise_time=T_swi, fall_time=T_swi,
                           pulse_width=5@u_ns, period=20@u_ns)
circuit.VoltageSource(3, 'b', circuit.gnd, V_dd)

# 定义门
circuit.X(1, gate, 'y', 'a', 'b', 'VDD', 'VSS')

# 进行瞬态仿真
simulator = circuit.simulator(temperature=27, nominal_temperature=27)
analysis = simulator.transient(step_time=0.0001 @u_ns, end_time=15 @u_ns)

# 计算延时
def calculate_propagation_delay(time, in_signal, out_signal, threshold):
    in_rise_times = []
    out_rise_times = []
    in_fall_times = []
    out_fall_times = []
    
    for i in range(1, len(time)):
        if in_signal[i-1] < threshold <= in_signal[i]:
            in_rise_times.append((time[i], in_signal[i]))
        if in_signal[i-1] > threshold >= in_signal[i]:
            in_fall_times.append((time[i], in_signal[i]))
        if out_signal[i-1] < threshold <= out_signal[i]:
            out_rise_times.append((time[i], out_signal[i]))
        if out_signal[i-1] > threshold >= out_signal[i]:
            out_fall_times.append((time[i], out_signal[i]))

    tpLH = None
    tpHL = None

    # 计算上升延迟
    if len(in_rise_times) > 0 and len(out_fall_times) > 0:
        tpLH = out_fall_times[0][0] - in_rise_times[0][0]
    
    # 计算下降延迟
    if len(in_fall_times) > 0 and len(out_rise_times) > 0:
        tpHL = out_rise_times[0][0] - in_fall_times[0][0]

    return tpLH, tpHL, in_rise_times, out_rise_times, in_fall_times, out_fall_times

tpLH, tpHL, in_rise_times, out_rise_times, in_fall_times, out_fall_times = calculate_propagation_delay(np.array(analysis.time), np.array(analysis['a']), np.array(analysis['y']), float(V_dd)/2)

print('tpLH:', tpLH, 's')
print('tpHL:', tpHL, 's')

def calculate_edge_times(time, signal, threshold_low, threshold_high):
    up_times_low = []
    up_times_high = []
    down_times_low = []
    down_times_high = []
    for i in range(1, len(time)):
        if signal[i-1] < threshold_low <= signal[i]:
            up_times_low.append((time[i], signal[i]))
        if signal[i-1] < threshold_high <= signal[i]:
            up_times_high.append((time[i], signal[i]))

    for i in range(1, len(time)):
        if signal[i] < threshold_low <= signal[i-1]:
            down_times_low.append((time[i], signal[i]))
        if signal[i] < threshold_high <= signal[i-1]:
            down_times_high.append((time[i], signal[i]))

    Trans_out_up = abs(up_times_low[0][0]-up_times_high[0][0])
    Trans_out_down = abs(down_times_low[0][0]-down_times_high[0][0])

    print('Up transition time:', Trans_out_up, 's')
    print('Down transition time:', Trans_out_down, 's')

    return up_times_low, up_times_high, down_times_low, down_times_high

up_times_low, up_times_high, down_times_low, down_times_high = calculate_edge_times(np.array(analysis.time), np.array(analysis['y']), float(V_dd)*0.05, float(V_dd)*0.95)

figure, ax = plt.subplots(figsize=(10, 6))
plot(analysis['a'], axis=ax, label='V(in1)')
plot(analysis['y'], axis=ax, label='V(out)')

# 标出变化的点
for t, v in in_rise_times:
    ax.plot(t, v, 'go', label='V(in1) up')  # 绿色表示输入上升沿
for t, v in out_rise_times:
    ax.plot(t, v, 'ro', label='V(out) up')  # 红色表示输出上升沿
for t, v in in_fall_times:
    ax.plot(t, v, 'bx', label='V(in1) down')  # 蓝色表示输入下降沿
for t, v in out_fall_times:
    ax.plot(t, v, 'mx', label='V(out) down')  # 品红色表示输出下降沿
for t, v in up_times_low:
    ax.plot(t, v, 'cs', label='V(out) up at 0.05*V_dd')  # 青色表示输出上升到0.01*V_dd
for t, v in up_times_high:
    ax.plot(t, v, 'ys', label='V(out) up at 0.95*V_dd')  # 黄色表示输出上升到0.99*V_dd
for t, v in down_times_low:
    ax.plot(t, v, 'rs', label='V(out) down at 0.05*V_dd')  # 青色表示输出上升到0.01*V_dd
for t, v in down_times_high:
    ax.plot(t, v, 'bs', label='V(out) down at 0.95*V_dd')  # 黄色表示输出上升到0.99*V_dd

plt.title(f'{gate} Gate Transient Analysis')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()
