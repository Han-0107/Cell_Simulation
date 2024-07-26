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
    parser.add_argument('--Cap_load_start', type=float, default=0.1, help='Start value of load capacitance')
    parser.add_argument('--Cap_load_end', type=float, default=0.5, help='End value of load capacitance')
    parser.add_argument('--Cap_load_step', type=float, default=0.1, help='Step value of load capacitance')
    parser.add_argument('--Trans_in_up', type=float, default=5.0e-09, help='UP value of transition time')
    parser.add_argument('--Trans_in_down', type=float, default=5.0e-09, help='Down value of transition time')
    parser.add_argument('--T_pulse', type=float, default=10, help='Value of pulse time')
    parser.add_argument('--T_period', type=float, default=40, help='Value of period time')
    parser.add_argument('--Vi_trans_up', type=float, nargs=5, default=[0.21, 0.63, 1.05, 1.47, 1.89], help='Voltages for up transitions at 0.1*V_dd, 0.3*Trans, 0.5*Trans, 0.7*Trans, 0.9*Trans')
    parser.add_argument('--Vi_trans_down', type=float, nargs=5, default=[1.89, 1.47, 1.05, 0.63, 0.21], help='Voltages for down transitions at 1.1*Trans+T_pulse, 1.3*Trans+T_pulse, 1.5*Trans+T_pulse, 1.7*Trans+T_pulse, 1.9*Trans+T_pulse')
    return parser.parse_args()

def define_variables(args):
    # 添加单位
    V_dd_start = args.V_dd_start @ u_V
    V_dd_end = args.V_dd_end @ u_V
    V_dd_step = args.V_dd_step @ u_V
    Cap_load_start = args.Cap_load_start @ u_pF
    Cap_load_end = args.Cap_load_end @ u_pF
    Cap_load_step = args.Cap_load_step @ u_pF
    Trans_in_up = args.Trans_in_up @ u_s
    Trans_in_down = args.Trans_in_down @ u_s
    T_pulse = args.T_pulse @ u_ns
    T_period = args.T_period @ u_ns
    Vi_trans_up = args.Vi_trans_up
    Vi_trans_down = args.Vi_trans_down

    # 生成变量范围
    V_dd_range = np.arange(V_dd_start, V_dd_end + V_dd_step, V_dd_step)
    Cap_load_range = np.arange(Cap_load_start, Cap_load_end + Cap_load_step, Cap_load_step)

    return V_dd_range, Cap_load_range, Trans_in_up, Trans_in_down, T_pulse, T_period, Vi_trans_up, Vi_trans_down

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

def create_circuit(V_dd, Cap_load, Trans_in_up, Trans_in_down, T_pulse, T_period, gate, Vi_trans_up, Vi_trans_down):
    # 创建电路
    circuit = Circuit('Gate for Experiment')

    # 包含 SPICE 模型库
    circuit.include('/home/yaohui/Research/PySpice/Libs/cells.sp')
    circuit.include('/home/yaohui/Research/PySpice/Libs/gpdk45nm.m')

    # 定义电源电压
    circuit.V(1, 'VDD', circuit.gnd, V_dd)
    circuit.V(2, 'VSS', circuit.gnd, 0 @ u_V)

    # 负载端的RC
    # circuit.R('out', 'y', 'out', R_out)
    circuit.C('out', 'y', circuit.gnd, Cap_load)

    circuit.PieceWiseLinearVoltageSource('Vpulse', 'a', circuit.gnd,
                                        values=[
                                                (0, 0), 
                                                (0.1*Trans_in_up, Vi_trans_up[0]),
                                                (0.3*Trans_in_up, Vi_trans_up[1]),
                                                (0.5*Trans_in_up, Vi_trans_up[2]),
                                                (0.7*Trans_in_up, Vi_trans_up[3]),
                                                (0.9*Trans_in_up, Vi_trans_up[4]),
                                                (Trans_in_up, V_dd), 
                                                (Trans_in_up+T_pulse, V_dd), 
                                                (0.1*Trans_in_down+Trans_in_up+T_pulse, Vi_trans_down[0]),
                                                (0.3*Trans_in_down+Trans_in_up+T_pulse, Vi_trans_down[1]),
                                                (0.5*Trans_in_down+Trans_in_up+T_pulse, Vi_trans_down[2]),
                                                (0.7*Trans_in_down+Trans_in_up+T_pulse, Vi_trans_down[3]),
                                                (0.9*Trans_in_down+Trans_in_up+T_pulse, Vi_trans_down[4]),                                                
                                                (Trans_in_down+Trans_in_up+T_pulse, 0), 
                                                (Trans_in_down+Trans_in_up+2*T_pulse, 0)
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

def get_voltage(time, signal, Trans_up, Trans_down, T_pulse):    # trans改成out的
    # 分别在这些时间节点记录电压值
    record_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    Vo_trans_up = [signal[np.where(time >= factor * Trans_up)[0][0]] for factor in record_factors]
    Vo_trans_down = [signal[np.where(time >= factor * Trans_down + Trans_up + T_pulse)[0][0]] for factor in record_factors]

    return Vo_trans_up, Vo_trans_down

def calculate_transition_times(time, signal, threshold_low, threshold_high):
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

    return Trans_out_up, Trans_out_down

def main():
    args = parse_arguments()
    V_dd_range, Cap_load_range, Trans_in_up, Trans_in_down, T_pulse, T_period, Vi_trans_up, Vi_trans_down = define_variables(args)

    results = []
    Vi_trans_up_floats = [float(value) for value in Vi_trans_up]
    Vi_trans_down_floats = [float(value) for value in Vi_trans_down]
    
    for V_dd in V_dd_range:
        for Cap_load in Cap_load_range:

                circuit = create_circuit(V_dd, Cap_load, Trans_in_up, Trans_in_down, T_pulse, T_period, args.gate, Vi_trans_up, Vi_trans_down)
                    
                # 进行瞬态仿真
                simulator = circuit.simulator(temperature=25, nominal_temperature=25)
                analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=45 @ u_ns)
                    
                # 计算延时
                tpLH, tpHL = calculate_propagation_delay(np.array(analysis.time), np.array(analysis['a']), np.array(analysis['y']), float(V_dd)/2, args.gate)
                tpLH = format(tpLH, '.4e') if tpLH is not None else 'None'
                tpHL = format(tpHL, '.4e') if tpHL is not None else 'None'
                    
                # 计算输出端transition time
                Trans_out_up, Trans_out_down = calculate_transition_times(np.array(analysis.time), np.array(analysis['y']), float(V_dd)*0.05, float(V_dd)*0.95)

                # 计算过渡电压
                Vo_trans_up, Vo_trans_down = get_voltage(np.array(analysis.time), np.array(analysis['a']), Trans_out_up, Trans_out_down, T_pulse)

                # 输出结果
                result = {
                            "gate": args.gate,
                            "V_dd": V_dd.value,
                            "Cap_load": Cap_load.value,
                            "Trans_in_up": Trans_in_up.value,
                            "Trans_in_down": Trans_in_down.value,
                            "Trans_out_up": Trans_out_up,
                            "Trans_out_down": Trans_out_down,
                            "Vi_trans_up": Vi_trans_up_floats,
                            "Vi_trans_down": Vi_trans_down_floats,
                            "Vo_trans_up": Vo_trans_up,
                            "Vo_trans_down": Vo_trans_down,
                            "tpLH": tpLH,
                            "tpHL": tpHL
                        }
                results.append(result)
                print(result)

    with open(f"./Results/{args.gate}_delay.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
