#!/usr/bin/env python3

import numpy as np
import sys
import os
import glob
from pathlib import Path

def recenter_snapshots(directory):
    # Convert to Path object for better path handling
    base_dir = Path(directory)
    
    # Create output directory if it doesn't exist
    output_dir = base_dir / "recentered"
    output_dir.mkdir(exist_ok=True)
    
    # Get all OUT files sorted numerically
    out_files = sorted(glob.glob(str(base_dir / "OUT.run0.*")), 
                      key=lambda x: int(x.split('.')[-1]))
    
    for in_file in out_files:
        if not in_file.split('.')[-1].isdigit():  # Skip checkpoint files
            continue
            
        # Create output filename
        out_file = output_dir / Path(in_file).name
        
        print(f"Processing {in_file} -> {out_file}...")
        
        # Read binary data
        with open(in_file, 'rb') as f:
            header = f.read(9656)  # Read header
            data = np.fromfile(f, dtype=np.float32)  # Read binary data
        
        particles = data.reshape((-1, 7))  # 7 columns per particle
        
        # Calculate total mass
        masses = particles[:, 0]
        total_mass = np.sum(masses)
        
        # Calculate center of mass position and velocity
        com_pos = np.sum(particles[:, 1:4] * masses[:, np.newaxis], axis=0) / total_mass
        com_vel = np.sum(particles[:, 4:7] * masses[:, np.newaxis], axis=0) / total_mass
        
        # Recenter positions and velocities
        particles[:, 1:4] -= com_pos
        particles[:, 4:7] -= com_vel
        
        # Save recentered data with original header
        with open(out_file, 'wb') as f:
            f.write(header)
            particles.astype(np.float32).tofile(f)
        
        print(f"  COM position before: {com_pos}")
        print(f"  COM velocity before: {com_vel}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recenter.py <simulation_directory>")
        sys.exit(1)
    
    recenter_snapshots(sys.argv[1])
    print("\nAll files processed and saved in 'recentered' subdirectory")
