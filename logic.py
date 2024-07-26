import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import *
from PySpice.Unit import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gate Simulation Parameters')
    parser.add_argument('--gate', type=str, default='NAND2X1', help='Gate type')
    parser.add_argument('--V_dd_start', type=float, default=2.1, help='Start value of voltage')
    parser.add_argument('--V_dd_end', type=float, default=2.4, help='End value of voltage')
    parser.add_argument('--V_dd_step', type=float, default=0.3, help='Step value of voltage')
    parser.add_argument('--R_out_start', type=float, default=1.0, help='Start value of load resistance')
    parser.add_argument('--R_out_end', type=float, default=1.1, help='End value of load resistance')
    parser.add_argument('--R_out_step', type=float, default=0.1, help='Step value of load resistance')
    parser.add_argument('--C_out_start', type=float, default=1.0, help='Start value of load capacitance')
    parser.add_argument('--C_out_end', type=float, default=1.1, help='End value of load capacitance')
    parser.add_argument('--C_out_step', type=float, default=0.1, help='Step value of load capacitance')
    parser.add_argument('--T_swi_start', type=float, default=5.0, help='Start value of transition time')
    parser.add_argument('--T_swi_end', type=float, default=6.0, help='End value of transition time')
    parser.add_argument('--T_swi_step', type=float, default=1.0, help='Step value of transition time')
    parser.add_argument('--T_pulse', type=float, default=10, help='Value of pulse time')
    parser.add_argument('--T_period', type=float, default=40, help='Value of period time')
    parser.add_argument('--Vl_trans_up', type=float, nargs=5, default=[0.21@ u_V, 0.63@ u_V, 1.05@ u_V, 1.47@ u_V, 1.89@ u_V], help='Voltages for up transitions at 0.1*V_dd, 0.3*T_swi, 0.5*T_swi, 0.7*T_swi, 0.9*T_swi')
    parser.add_argument('--Vl_trans_down', type=float, nargs=5, default=[1.89@ u_V, 1.47@ u_V, 1.05@ u_V, 0.63@ u_V, 0.21@ u_V], help='Voltages for down transitions at 1.1*T_swi+T_pulse, 1.3*T_swi+T_pulse, 1.5*T_swi+T_pulse, 1.7*T_swi+T_pulse, 1.9*T_swi+T_pulse')
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
    T_pulse = args.T_pulse @ u_ns
    T_period = args.T_period @ u_ns
    Vl_trans_up = args.Vl_trans_up
    Vl_trans_down = args.Vl_trans_down

    # 生成变量范围
    V_dd_range = np.arange(V_dd_start, V_dd_end + V_dd_step, V_dd_step)
    R_out_range = np.arange(R_out_start, R_out_end + R_out_step, R_out_step)
    C_out_range = np.arange(C_out_start, C_out_end + C_out_step, C_out_step)
    T_swi_range = np.arange(T_swi_start, T_swi_end + T_swi_step, T_swi_step)

    return V_dd_range, R_out_range, C_out_range, T_swi_range, T_pulse, T_period, Vl_trans_up, Vl_trans_down

def calculate_propagation_delay(time, in_signal, out_signal, threshold, gate):
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

    if gate == 'INVX1' or gate == 'NAND2X1' or gate == 'NOR2X1' or gate == 'XNOR2X1': # 还要加上其他逻辑门！
        # 计算上升延迟
        if len(in_rise_times) > 0 and len(out_fall_times) > 0:
            tpLH = out_fall_times[0][0] - in_rise_times[0][0]
    
        # 计算下降延迟
        if len(in_fall_times) > 0 and len(out_rise_times) > 0:
            tpHL = out_rise_times[0][0] - in_fall_times[0][0]
    else:
        # 计算上升延迟
        if len(in_rise_times) > 0 and len(out_rise_times) > 0:
            tpLH = out_rise_times[0][0] - in_rise_times[0][0]
    
        # 计算下降延迟
        if len(in_fall_times) > 0 and len(out_fall_times) > 0:
            tpHL = out_fall_times[0][0] - in_fall_times[0][0]

    return tpLH, tpHL

def create_circuit(V_dd, R_out, C_out, T_swi, T_pulse, T_period, gate, Vl_trans_up, Vl_trans_down):
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

    circuit.PieceWiseLinearVoltageSource('Vpulse', 'a', circuit.gnd,
                                        values=[
                                                (0, 0), 
                                                (0.1*T_swi, Vl_trans_up[0]),
                                                (0.3*T_swi, Vl_trans_up[1]),
                                                (0.5*T_swi, Vl_trans_up[2]),
                                                (0.7*T_swi, Vl_trans_up[3]),
                                                (0.9*T_swi, Vl_trans_up[4]),
                                                (T_swi, V_dd), 
                                                (T_swi+T_pulse, V_dd), 
                                                (1.1*T_swi+T_pulse, Vl_trans_down[0]),
                                                (1.3*T_swi+T_pulse, Vl_trans_down[1]),
                                                (1.5*T_swi+T_pulse, Vl_trans_down[2]),
                                                (1.7*T_swi+T_pulse, Vl_trans_down[3]),
                                                (1.9*T_swi+T_pulse, Vl_trans_down[4]),                                                
                                                (2*T_swi+T_pulse, 0), 
                                                (2*T_swi+T_period, 0)
                                                ]
                                        )
    
    if gate == 'NAND2X1' or gate == 'AND2X1':
        circuit.VoltageSource(3, 'b', circuit.gnd, V_dd)
        circuit.X(1, gate, 'y', 'a', 'b', 'VDD', 'VSS')
    elif gate == 'INVX1':
        circuit.X(1, gate, 'y', 'a', 'VDD', 'VSS')
    else:
        circuit.VoltageSource(3, 'b', circuit.gnd, 0 @ u_V)
        circuit.X(1, gate, 'y', 'a', 'b', 'VDD', 'VSS')

    return circuit

def get_voltage(time, signal, T_swi, T_pulse):
    # 分别在这些时间节点记录电压值
    up_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    down_factors = [1.1, 1.3, 1.5, 1.7, 1.9]
    
    Vc_trans_up = [signal[np.where(time >= factor * T_swi)[0][0]] for factor in up_factors]
    Vc_trans_down = [signal[np.where(time >= factor * T_swi + T_pulse)[0][0]] for factor in down_factors]
    
    return Vc_trans_up, Vc_trans_down

def main():
    args = parse_arguments()
    V_dd_range, R_out_range, C_out_range, T_swi_range, T_pulse, T_period, Vl_trans_up, Vl_trans_down = define_variables(args)

    results = []
    Vl_trans_up_floats = [float(value) for value in Vl_trans_up]
    Vl_trans_down_floats = [float(value) for value in Vl_trans_down]
    
    for V_dd in V_dd_range:
        for R_out in R_out_range:
            for C_out in C_out_range:
                for T_swi in T_swi_range:

                    circuit = create_circuit(V_dd, R_out, C_out, T_swi, T_pulse, T_period, args.gate, Vl_trans_up, Vl_trans_down)
                    
                    # 进行瞬态仿真
                    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
                    analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=45 @ u_ns)
                    
                    # 计算延时
                    tpLH, tpHL = calculate_propagation_delay(np.array(analysis.time), np.array(analysis['a']), np.array(analysis['y']), float(V_dd)/2, args.gate)
                    tpLH = format(tpLH, '.4e') if tpLH is not None else 'None'
                    tpHL = format(tpHL, '.4e') if tpHL is not None else 'None'
                    
                    # 计算过渡电压
                    Vc_trans_up, Vc_trans_down = get_voltage(np.array(analysis.time), np.array(analysis['a']), T_swi, T_pulse)

                    # 输出结果
                    result = {
                        "gate": args.gate,
                        "V_dd": V_dd.value,
                        "R_out": R_out.value,
                        "C_out": C_out.value,
                        "T_swi": T_swi.value,
                        "Vl_trans_up": Vl_trans_up_floats,
                        "Vl_trans_down": Vl_trans_down_floats,
                        "Vc_trans_up": Vc_trans_up,
                        "Vc_trans_down": Vc_trans_down,
                        "tpLH": tpLH,
                        "tpHL": tpHL
                    }
                    results.append(result)
                    print(result)

    with open(f"./Results/{args.gate}_delay.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
