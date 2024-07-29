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
    parser.add_argument('--gate', type=str, default='NOR2X1', help='Gate type')
    parser.add_argument('--V_dd', type=float, default=1.1, help='Voltage')
    parser.add_argument('--Cap_load', type=float, default=0.1, help='Load capacitance')
    parser.add_argument('--Trans_in_up', type=float, default=5, help='UP time of transition')
    parser.add_argument('--Trans_in_down', type=float, default=5, help='Down time of transition')
    parser.add_argument('--T_pulse', type=float, default=5, help='Time of pulse')
    parser.add_argument('--T_period', type=float, default=20, help='Time of period')
    parser.add_argument('--Vi_trans_up', type=float, nargs=5, default=[0.11, 0.33, 0.55, 0.77, 0.99], help='Voltages for up transitions')
    parser.add_argument('--Vi_trans_down', type=float, nargs=5, default=[0.99, 0.77, 0.55, 0.33, 0.11], help='Voltages for down transition')
    return parser.parse_args()

def define_variables(args):
    # 添加单位
    V_dd = args.V_dd @ u_V
    Cap_load = args.Cap_load @ u_pF
    Trans_in_up = args.Trans_in_up @ u_ns
    Trans_in_down = args.Trans_in_down @ u_ns
    T_pulse = args.T_pulse @ u_ns
    T_period = args.T_period @ u_ns
    Vi_trans_up = args.Vi_trans_up
    Vi_trans_down = args.Vi_trans_down

    return V_dd, Cap_load, Trans_in_up, Trans_in_down, T_pulse, T_period, Vi_trans_up, Vi_trans_down

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

    return tpLH, tpHL, in_rise_times, out_rise_times, in_fall_times, out_fall_times

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

def get_voltage(time, signal, T_period):
    
    # 提取仿真采样电压值
    record_factors = np.arange(0, 1, 0.01).tolist()
    max_time = T_period
    Vo = [round(abs(signal[np.where(time >= factor * max_time)[0][0]]), 4) for factor in record_factors]
    
    return Vo

def resize_list(original_list, target_length):
    original_length = len(original_list)
    if original_length == target_length:
        return original_list
    
    # 使用线性插值进行压缩或扩展
    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)
    resized_list = np.interp(target_indices, original_indices, original_list)
    return resized_list.tolist()

def extract_segments(data, threshold=0.03, min_diff=0.003): # 采样阈值
    def filter_segment(segment):
        filtered_segment = []
        last_value = None
        for value in segment:
            if last_value is None or abs(value - last_value) >= min_diff:
                filtered_segment.append(value)
                last_value = value
        return filtered_segment
    
    rising_segment = []
    falling_segment = []
    
    rising = None
    
    for i in range(1, len(data)):
        if rising is None:
            if data[i] - data[i-1] > threshold:
                rising = True
            elif data[i-1] - data[i] > threshold:
                rising = False
        
        if rising:
            if data[i] - data[i-1] >= -threshold:
                if not rising_segment or data[i-1] != rising_segment[-1]:
                    rising_segment.append(data[i-1])
            else:
                if not rising_segment or data[i-1] != rising_segment[-1]:
                    rising_segment.append(data[i-1])
                rising = False
                if not falling_segment or data[i] != falling_segment[-1]:
                    falling_segment.append(data[i])
        if not rising:
            if data[i-1] - data[i] >= -threshold:
                if not falling_segment or data[i-1] != falling_segment[-1]:
                    falling_segment.append(data[i-1])
            else:
                if not falling_segment or data[i-1] != falling_segment[-1]:
                    falling_segment.append(data[i-1])
                rising = True
                if not rising_segment or data[i] != rising_segment[-1]:
                    rising_segment.append(data[i])
    
    if rising:
        if not rising_segment or data[-1] != rising_segment[-1]:
            rising_segment.append(data[-1])
    else:
        if not falling_segment or data[-1] != falling_segment[-1]:
            falling_segment.append(data[-1])
    
    rising_segment = filter_segment(rising_segment)
    falling_segment = filter_segment(falling_segment)
    falling_segment = falling_segment[2:]   # 经验之谈
    
    return rising_segment, falling_segment

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

    return Trans_out_up, Trans_out_down, up_times_low, up_times_high, down_times_low, down_times_high

def main():
    args = parse_arguments()
    V_dd, Cap_load, Trans_in_up, Trans_in_down, T_pulse, T_period, Vi_trans_up, Vi_trans_down = define_variables(args)

    results = []
    Vi_trans_up_floats = [float(value) for value in Vi_trans_up]
    Vi_trans_down_floats = [float(value) for value in Vi_trans_down]


    circuit = create_circuit(V_dd, Cap_load, Trans_in_up, Trans_in_down, T_pulse, T_period, args.gate, Vi_trans_up, Vi_trans_down)
                    
    # 进行瞬态仿真
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=25 @ u_ns)
                    
    # 计算延时
    tpLH, tpHL, in_rise_times, out_rise_times, in_fall_times, out_fall_times = calculate_propagation_delay(np.array(analysis.time), np.array(analysis['a']), np.array(analysis['y']), float(V_dd)/2, args.gate)
    tpLH = format(tpLH, '.4e') if tpLH is not None else 'None'
    tpHL = format(tpHL, '.4e') if tpHL is not None else 'None'
                    
    # 计算输出端transition time
    Trans_out_up, Trans_out_down, up_times_low, up_times_high, down_times_low, down_times_high = calculate_transition_times(np.array(analysis.time), np.array(analysis['y']), float(V_dd)*0.05, float(V_dd)*0.95)

    # 计算过渡电压
    Vout = get_voltage(np.array(analysis.time), np.array(analysis['y']), T_period)
    Vout_up, Vout_down = extract_segments(Vout)

    len_vin = len(Vi_trans_up)
    Vout_up = resize_list(Vout_up, len_vin)
    Vout_down = resize_list(Vout_down, len_vin)

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
                "Vout_up": Vout_up,
                "Vout_down": Vout_down,
                "tpLH": tpLH,
                "tpHL": tpHL
            }
    results.append(result)
    print(result)

    with open(f"./Results/{args.gate}_delay.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)

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

    plt.title(f"{args.gate} Gate Transient Analysis")
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
