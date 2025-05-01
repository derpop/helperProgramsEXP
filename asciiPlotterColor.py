#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Output directory
output_dir = "particlePositionsColor"
os.makedirs(output_dir, exist_ok=True)

# Input filename base
base_filename = "OUTASC.run0."

# Total number of files
num_files = 550

# Plot settings
max_radius = 50.0   # Clip particles farther than this
dpi_setting = 300   # Save DPI
alpha_val = 0.5     # Transparency
cmap_choice = 'inferno'  # Color map for z-depth

# Helper function to process a single file
def process_file(i):
    filename = f"{base_filename}{i:05d}"

    if not os.path.isfile(filename):
        print(f"File {filename} not found, skipping.")
        return

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
                # Extract x, y, z
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                positions.append((x, y, z))
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    if len(positions) == 0:
        print(f"No particle data in {filename}, skipping.")
        return

    positions = np.array(positions)

    # Cut by 3D radius
    radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    positions = positions[radii <= max_radius]

    if len(positions) == 0:
        print(f"All particles outside radius in {filename}, skipping plot.")
        return

    # Extract z for color
    z_values = positions[:, 2]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1],
                c=z_values, cmap=cmap_choice,
                s=0.4, alpha=alpha_val, edgecolors='none')
    plt.xlabel('x')
    plt.ylabel('y')
    if time_value is not None:
        plt.title(f'Particle Positions (Time={time_value:.3f})')
    else:
        plt.title('Particle Positions (Unknown Time)')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.axis('equal')
    plt.grid(False)

    # Save figure
    if time_value is not None:
        output_path = os.path.join(output_dir, f"positions_{i:05d}_t{time_value:.3f}.png")
    else:
        output_path = os.path.join(output_dir, f"positions_{i:05d}.png")

    try:
        plt.savefig(output_path, dpi=dpi_setting)
        plt.close()
        print(f"Saved {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

# ----------
# Main Run
# ----------
if __name__ == "__main__":
    num_workers = 10  # Adjust if needed
    indices = list(range(num_files))
    with Pool(processes=num_workers) as pool:
        pool.map(process_file, indices)
    print("✅ All plots finished.")
