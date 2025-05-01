import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_model_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        if line.strip().startswith("!") or len(line.strip()) == 0:
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                float_vals = list(map(float, parts[:4]))
                data_lines.append(float_vals)
            except ValueError:
                continue

    return np.array(data_lines)

def plot_columns(data, file_path):
    R = data[:, 0]
    D = data[:, 1]
    M = data[:, 2]
    P = data[:, 3]

    base = os.path.splitext(os.path.basename(file_path))[0]

    plt.figure()
    plt.plot(R, D, label='D vs R')
    plt.xlabel('R')
    plt.ylabel('D')
    plt.title('D vs R')
    plt.grid(True)
    plt.savefig(f'{base}_D_vs_R.png')
    plt.close()

    plt.figure()
    plt.plot(R, M, label='M vs R', color='green')
    plt.xlabel('R')
    plt.ylabel('M')
    plt.title('M vs R')
    plt.grid(True)
    plt.savefig(f'{base}_M_vs_R.png')
    plt.close()

    plt.figure()
    plt.plot(R, P, label='P vs R', color='red')
    plt.xlabel('R')
    plt.ylabel('P')
    plt.title('P vs R')
    plt.grid(True)
    plt.savefig(f'{base}_P_vs_R.png')
    plt.close()

    # Optional: all on one plot
    plt.figure()
    plt.plot(R, D, label='D', alpha=0.8)
    plt.plot(R, M, label='M', alpha=0.8)
    plt.plot(R, P, label='P', alpha=0.8)
    plt.xlabel('R')
    plt.ylabel('Value')
    plt.title('D, M, P vs R')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{base}_All_vs_R.png')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 modelAnalysis.py <model_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_model_file(file_path)
    plot_columns(data, file_path)

