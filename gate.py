import numpy as np
import argparse
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import *
from PySpice.Unit import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gate Simulation Parameters')
    parser.add_argument('--gate', type=str, required=True, help='Gate type')
    parser.add_argument('--V_dd_start', type=float, default=1.8, help='Start value of V_dd')
    parser.add_argument('--V_dd_end', type=float, default=2.2, help='End value of V_dd')
    parser.add_argument('--V_dd_step', type=float, default=0.2, help='Step value of V_dd')
    parser.add_argument('--R_out_start', type=float, default=5.0, help='Start value of R_out')
    parser.add_argument('--R_out_end', type=float, default=15.0, help='End value of R_out')
    parser.add_argument('--R_out_step', type=float, default=5.0, help='Step value of R_out')
    parser.add_argument('--C_out_start', type=float, default=0.05, help='Start value of C_out')
    parser.add_argument('--C_out_end', type=float, default=0.15, help='End value of C_out')
    parser.add_argument('--C_out_step', type=float, default=0.05, help='Step value of C_out')
    parser.add_argument('--T_swi_start', type=float, default=0.5, help='Start value of T_swi')
    parser.add_argument('--T_swi_end', type=float, default=1.5, help='End value of T_swi')
    parser.add_argument('--T_swi_step', type=float, default=0.5, help='Step value of T_swi')

    return parser.parse_args()

def define_variables(args):
    # 添加单位
    V_dd_start = args.V_dd_start @ u_V
    V_dd_end = args.V_dd_end @ u_V
    V_dd_step = args.V_dd_step @ u_V
    R_out_start = args.R_out_start @ u_Ohm
    R_out_end = args.R_out_end @ u_Ohm
    R_out_step = args.R_out_step @ u_Ohm
    C_out_start = args.C_out_start @ u_pF
    C_out_end = args.C_out_end @ u_pF
    C_out_step = args.C_out_step @ u_pF
    T_swi_start = args.T_swi_start @ u_ns
    T_swi_end = args.T_swi_end @ u_ns
    T_swi_step = args.T_swi_step @ u_ns

    # 生成变量范围
    V_dd_range = np.arange(V_dd_start, V_dd_end + V_dd_step, V_dd_step)
    R_out_range = np.arange(R_out_start, R_out_end + R_out_step, R_out_step)
    C_out_range = np.arange(C_out_start, C_out_end + C_out_step, C_out_step)
    T_swi_range = np.arange(T_swi_start, T_swi_end + T_swi_step, T_swi_step)

    return V_dd_range, R_out_range, C_out_range, T_swi_range

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

def create_circuit(V_dd, R_out, C_out, T_swi, Gate):
    # 创建电路
    circuit = Circuit('Gate for Experiment')

    # 包含 SPICE 模型库
    circuit.include('/home/yaohui/Research/PySpice/Libs/cells.sp')
    circuit.include('/home/yaohui/Research/PySpice/Libs/gpdk45nm.m')

    # 定义电源电压
    circuit.V(1, 'VDD', circuit.gnd, V_dd)
    circuit.V(2, 'VSS', circuit.gnd, 0 @ u_V)

    # 负载端的RC
    circuit.R('out', 'y', 'out', R_out)
    circuit.C('out', 'out', circuit.gnd, C_out)

    # 定义输入信号（将PULSE信号改为PWL方法，给出信号在不同时间点的信息）
    circuit.PulseVoltageSource('Vpulse', 'a', circuit.gnd, initial_value=0 @ u_V, pulsed_value=V_dd, 
                               delay_time=0 @ u_ns, rise_time=T_swi, fall_time=T_swi, pulse_width=20 @ u_ns, period=40 @ u_ns)
    
    if Gate == 'NAND2X1' or Gate == 'AND2X1':
        circuit.VoltageSource(3, 'b', circuit.gnd, V_dd)
        circuit.X(1, Gate, 'y', 'a', 'b', 'VDD', 'VSS')
    elif Gate == 'INVX1':
        circuit.X(1, Gate, 'y', 'a', 'VDD', 'VSS')
    else:
        circuit.VoltageSource(3, 'b', circuit.gnd, 0 @ u_V)
        circuit.X(1, Gate, 'y', 'a', 'b', 'VDD', 'VSS')

    return circuit

def main():
    args = parse_arguments()
    V_dd_range, R_out_range, C_out_range, T_swi_range = define_variables(args)

    with open(f"./Results/{args.gate}_delay.txt", 'w') as file:
        for V_dd in V_dd_range:
            for R_out in R_out_range:
                for C_out in C_out_range:
                    for T_swi in T_swi_range:

                        circuit = create_circuit(V_dd, R_out, C_out, T_swi, args.gate)
                        
                        # 进行瞬态仿真
                        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
                        analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=100 @ u_ns)
                        
                        # 计算延时
                        tpLH, tpHL = calculate_delay(np.array(analysis.time), np.array(analysis['y']), float(V_dd)/2)
                        tpLH = format(tpLH, '.4e') if tpLH is not None else 'None'
                        tpHL = format(tpHL, '.4e') if tpHL is not None else 'None'
                        
                        # 输出结果
                        file.write(f"{args.gate}, {V_dd.value:.2f}, {R_out.value:.2f}, {C_out.value:.2f}, {T_swi.value:.2f}, {tpLH}, {tpHL}\n")
                        print(f"{args.gate}, {V_dd.value:.2f}, {R_out.value:.2f}, {C_out.value:.2f}, {T_swi.value:.2f}, {tpLH}, {tpHL}")

if __name__ == "__main__":
    main()
