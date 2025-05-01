import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Output directory
output_dir = "particlePositionsHalo"
os.makedirs(output_dir, exist_ok=True)

# Input filename base
base_filename = "outasciiHalo."

# Total number of files
num_files = 4000

# Plot settings
max_radius = 10.0  # Clip particles farther than this
dpi_setting = 150  # Save DPI (lower for faster saving)

# Helper function: handles ONE file
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

    # Cut by radius
    r3d = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    r2d = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    mask = (r3d <= 50.0) & (r2d <= 20.0)
    positions = positions[mask]


    if len(positions) == 0:
        print(f"All particles outside radius in {filename}, skipping plot.")
        return

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=0.3, alpha=0.2, edgecolors='none')
    plt.xlabel('x')
    plt.ylabel('y')
    if time_value is not None:
        plt.title(f'Particle Positions (Time={time_value:.3f})')
    else:
        plt.title('Particle Positions (Unknown Time)')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.axis('equal')
    plt.grid(True)

    # Save plot
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

# ---------
# Main run
# ---------

if __name__ == "__main__":
    # Number of workers: since you're on a shared machine, be careful
    num_workers = 12  # <--- Change if needed (4 threads is polite on shared machine)

    # List of indices to process
    indices = list(range(num_files))

    # Launch multiprocessing Pool
    with Pool(processes=num_workers) as pool:
        pool.map(process_file, indices)

    print("✅ All plots finished.")
