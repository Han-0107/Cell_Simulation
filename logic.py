import numpy as np
import argparse
import json
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import *
from PySpice.Unit import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gate Simulation Parameters')
    parser.add_argument('--gate', type=str, required=True, help='Gate type')
    parser.add_argument('--V_dd_start', type=float, default=1.8, help='Start value of voltage')
    parser.add_argument('--V_dd_end', type=float, default=2.2, help='End value of voltage')
    parser.add_argument('--V_dd_step', type=float, default=0.4, help='Step value of voltage')
    parser.add_argument('--R_out_start', type=float, default=5.0, help='Start value of load resistance')
    parser.add_argument('--R_out_end', type=float, default=15.0, help='End value of load resistance')
    parser.add_argument('--R_out_step', type=float, default=10.0, help='Step value of load resistance')
    parser.add_argument('--C_out_start', type=float, default=0.05, help='Start value of load capacitance')
    parser.add_argument('--C_out_end', type=float, default=0.15, help='End value of load capacitance')
    parser.add_argument('--C_out_step', type=float, default=0.1, help='Step value of load capacitance')
    parser.add_argument('--T_swi_start', type=float, default=0.5, help='Start value of transition time')
    parser.add_argument('--T_swi_end', type=float, default=1.5, help='End value of transition time')
    parser.add_argument('--T_swi_step', type=float, default=1.0, help='Step value of transition time')
    parser.add_argument('--T_pulse', type=float, default=20, help='Value of pulse time')
    parser.add_argument('--T_period', type=float, default=40, help='Value of period time')
    parser.add_argument('--V_1_up', type=float, default=0.18, help='Voltage when V=0.1*V_dd')
    parser.add_argument('--V_3_up', type=float, default=0.54, help='Voltage when 0.3*T_swi')
    parser.add_argument('--V_5_up', type=float, default=0.9, help='Voltage when 0.5*T_swi')
    parser.add_argument('--V_7_up', type=float, default=1.26, help='Voltage when 0.7*T_swi')
    parser.add_argument('--V_9_up', type=float, default=1.62, help='Voltage when 0.9*T_swi')
    parser.add_argument('--V_1_down', type=float, default=1.62, help='Voltage when 1.1*T_swi+T_pulse')
    parser.add_argument('--V_3_down', type=float, default=1.26, help='Voltage when 1.3*T_swi+T_pulse')
    parser.add_argument('--V_5_down', type=float, default=0.9, help='Voltage when 1.5*T_swi+T_pulse')
    parser.add_argument('--V_7_down', type=float, default=0.54, help='Voltage when 1.7*T_swi+T_pulse')
    parser.add_argument('--V_9_down', type=float, default=0.18, help='Voltage when 1.9*T_swi+T_pulse')
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
    V_1_up = args.V_1_up @ u_V
    V_3_up = args.V_3_up @ u_V
    V_5_up = args.V_5_up @ u_V
    V_7_up = args.V_7_up @ u_V
    V_9_up = args.V_9_up @ u_V
    V_1_down = args.V_1_down @ u_V
    V_3_down = args.V_3_down @ u_V
    V_5_down = args.V_5_down @ u_V
    V_7_down = args.V_7_down @ u_V
    V_9_down = args.V_9_down @ u_V
    Vl_trans_up = [V_1_up, V_3_up, V_5_up, V_7_up, V_9_up]
    Vl_trans_down = [V_1_down, V_3_down, V_5_down, V_7_down, V_9_down]

    # 生成变量范围
    V_dd_range = np.arange(V_dd_start, V_dd_end + V_dd_step, V_dd_step)
    R_out_range = np.arange(R_out_start, R_out_end + R_out_step, R_out_step)
    C_out_range = np.arange(C_out_start, C_out_end + C_out_step, C_out_step)
    T_swi_range = np.arange(T_swi_start, T_swi_end + T_swi_step, T_swi_step)

    return V_dd_range, R_out_range, C_out_range, T_swi_range, T_pulse, T_period, Vl_trans_up, Vl_trans_down

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

def create_circuit(V_dd, R_out, C_out, T_swi, T_pulse, T_period, Gate, Vl_trans_up, Vl_trans_down):
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

    V_1_up = Vl_trans_up[0]
    V_3_up = Vl_trans_up[1]
    V_5_up = Vl_trans_up[2]
    V_7_up = Vl_trans_up[3]
    V_9_up = Vl_trans_up[4]
    V_1_down = Vl_trans_down[0]
    V_3_down = Vl_trans_down[1]
    V_5_down = Vl_trans_down[2]
    V_7_down = Vl_trans_down[3]
    V_9_down = Vl_trans_down[4]

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
    
    if Gate == 'NAND2X1' or Gate == 'AND2X1':
        circuit.VoltageSource(3, 'b', circuit.gnd, V_dd)
        circuit.X(1, Gate, 'y', 'a', 'b', 'VDD', 'VSS')
    elif Gate == 'INVX1':
        circuit.X(1, Gate, 'y', 'a', 'VDD', 'VSS')
    else:
        circuit.VoltageSource(3, 'b', circuit.gnd, 0 @ u_V)
        circuit.X(1, Gate, 'y', 'a', 'b', 'VDD', 'VSS')

    return circuit

def calculate_time(time, signal, T_swi, T_pulse):
    Vc_1_up = np.where(time >= 0.1*T_swi)[0][0]
    Vc_3_up = np.where(time >= 0.3*T_swi)[0][0]
    Vc_5_up = np.where(time >= 0.5*T_swi)[0][0]
    Vc_7_up = np.where(time >= 0.7*T_swi)[0][0]
    Vc_9_up = np.where(time >= 0.9*T_swi)[0][0]
    Vc_1_down = np.where(time >= 1.1*T_swi+T_pulse)[0][0]
    Vc_3_down = np.where(time >= 1.3*T_swi+T_pulse)[0][0]
    Vc_5_down = np.where(time >= 1.5*T_swi+T_pulse)[0][0]
    Vc_7_down = np.where(time >= 1.7*T_swi+T_pulse)[0][0]
    Vc_9_down = np.where(time >= 1.9*T_swi+T_pulse)[0][0]
    Vc_trans_up = [signal[Vc_1_up], signal[Vc_3_up], signal[Vc_5_up], signal[Vc_7_up], signal[Vc_9_up]]
    Vc_trans_down = [signal[Vc_1_down], signal[Vc_3_down], signal[Vc_5_down], signal[Vc_7_down], signal[Vc_9_down]]
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
                    analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=100 @ u_ns)
                    
                    # 计算延时
                    tpLH, tpHL = calculate_delay(np.array(analysis.time), np.array(analysis['y']), float(V_dd)/2)
                    tpLH = format(tpLH, '.4e') if tpLH is not None else 'None'
                    tpHL = format(tpHL, '.4e') if tpHL is not None else 'None'
                    
                    # 计算过渡电压
                    Vc_trans_up, Vc_trans_down = calculate_time(np.array(analysis.time), np.array(analysis['a']), T_swi, T_pulse)

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
