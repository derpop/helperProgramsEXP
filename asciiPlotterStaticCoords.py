#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate particle position plots.')
    parser.add_argument('base_filename', type=str, 
                       help='Base filename for input files (e.g., "OUTASC.run0.")')
    return parser.parse_args()

# Configuration
args = parse_arguments()
base_filename = args.base_filename
output_dir = f"OUT_PLOTS_{base_filename.rstrip('.')}"
os.makedirs(output_dir, exist_ok=True)
num_files = 550
max_radius = 50.0
dpi_setting = 300
alpha_val = 0.5
cmap_choice = 'inferno'

def read_positions(i):
    filename = f"{base_filename}{i:05d}"
    if not os.path.isfile(filename):
        return None, None

    positions = []
    time_value = None

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    if 'Time=' in line:
                        try:
                            time_value = float(line.split('Time=')[1])
                        except ValueError:
                            pass
                    continue
                if line == '':
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                r = np.sqrt(x**2 + y**2 + z**2)
                if r <= max_radius:
                    positions.append((x, y, z))
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

    if len(positions) == 0:
        return None, time_value

    return np.array(positions), time_value

def find_first_valid_bounds():
    for i in range(num_files):
        positions, _ = read_positions(i)
        if positions is not None and len(positions) > 0:
            min_x = np.min(positions[:, 0])
            max_x = np.max(positions[:, 0])
            min_y = np.min(positions[:, 1])
            max_y = np.max(positions[:, 1])
            return min_x, max_x, min_y, max_y
    raise RuntimeError("No valid frames found to extract axis bounds.")

def process_file(i, min_x, max_x, min_y, max_y):
    filename = f"{base_filename}{i:05d}"
    if not os.path.isfile(filename):
        print(f"File {filename} not found, skipping.")
        return

    positions, time_value = read_positions(i)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Plot data if available
    if positions is not None and len(positions) > 0:
        z_values = positions[:, 2]
        sc = ax.scatter(positions[:, 0], positions[:, 1],
                       c=z_values, cmap=cmap_choice,
                       s=0.4, alpha=alpha_val, edgecolors='none')

    # Enforce limits FIRST and FOREMOST
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Calculate and set the correct aspect ratio SECOND
    x_range = max_x - min_x
    y_range = max_y - min_y
    ax.set_aspect(x_range/y_range)

    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if time_value is not None:
        ax.set_title(f'Particle Positions (Time={time_value:.3f})')
    else:
        ax.set_title('Particle Positions (Unknown Time)')
    ax.grid(False)

    output_filename = (f"positions_{i:05d}_t{time_value:.3f}.png"
                      if time_value is not None else f"positions_{i:05d}.png")
    output_path = os.path.join(output_dir, output_filename)

    try:
        plt.savefig(output_path, dpi=dpi_setting, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    print("🔍 Scanning for global axis limits...")
    global_min_x, global_max_x, global_min_y, global_max_y = find_first_valid_bounds()
    print(f"📏 Global bounds found:")
    print(f"x: {global_min_x:.2f} to {global_max_x:.2f}")
    print(f"y: {global_min_y:.2f} to {global_max_y:.2f}")

    print("🖼️   Generating plots...")
    with Pool(processes=10) as pool:
        pool.map(partial(process_file,
                        min_x=global_min_x,
                        max_x=global_max_x,
                        min_y=global_min_y,
                        max_y=global_max_y),
                range(num_files))
    print("✅ All plots finished.")
