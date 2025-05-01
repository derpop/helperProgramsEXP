#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

#FILEPATH INCORPORATED INTO TERMINAL ARGUMENTS
#file_path = 'MWhalo.bods'  # Update this with your .bods file path

# Function to load .bods file data
def load_bods_file(file_path):
    with open(file_path, 'r') as file:
        data_lines = file.readlines()
    
    # Parse position (x, y, z) and velocity (vx, vy, vz) data, skipping the header
    data = [list(map(float, line.split()[1:7])) for line in data_lines[1:]]
    return np.array(data)

# Function to analyze spherical symmetry and calculate actual density profile
def analyze_halo_data_density(data):
    # Extract positions
    positions = data[:, :3]
    
    # Calculate radii
    radii = np.linalg.norm(positions, axis=1)
    
    # Define bins for the radii
    bins = np.linspace(0, np.max(radii), 50)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Count particles in each spherical shell
    counts, _ = np.histogram(radii, bins=bins)
    
    # Calculate volume of each spherical shell
    shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    
    # Calculate density (particles per unit volume)
    densities = counts / shell_volumes
    
    # Plot the actual density profile
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, densities, 'o-', color='blue')
    plt.xlabel('Radius (r)')
    plt.ylabel('Density')
    plt.title('Actual Density Profile of Bodies')
    plt.yscale('log')  # Using log scale to better visualize density drop-off
    plt.grid(True)
    plt.savefig('ActualDensityProfile.png')
    plt.close()

# Function to analyze spherical symmetry and velocity distribution
def analyze_halo_data(data):
    # Extract positions and velocities
    positions = data[:, :3]
    velocities = data[:, 3:]
    
    # Calculate spherical coordinates
    radii = np.linalg.norm(positions, axis=1)
    theta = np.arccos(positions[:, 2] / (radii + 1e-10))  # Avoid division by zero
    phi = np.arctan2(positions[:, 1], positions[:, 0])
    
    # Calculate velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)

    # Calculate average and max values
    avg_radius = np.mean(radii)
    max_radius = np.max(radii)
    avg_velocity = np.mean(velocity_magnitudes)
    max_velocity = np.max(velocity_magnitudes)
    
    print(f"Average Radius: {avg_radius:.2f}, Max Radius: {max_radius:.2f}")
    print(f"Average Velocity: {avg_velocity:.2f}, Max Velocity: {max_velocity:.2f}")

    # Plot radial distribution
    plt.figure(figsize=(8, 6))
    plt.hist(radii, bins=50, density=True, alpha=0.7, color='blue')
    plt.xlabel('Radius (r)')
    plt.ylabel('Density')
    plt.title('Radial Distribution of Points')
    plt.grid(True)
    plt.savefig('RadialDistribution.png')
    plt.close()

    # Plot theta (polar angle) distribution
    plt.figure(figsize=(8, 6))
    plt.hist(theta, bins=50, density=True, alpha=0.7, color='green')
    plt.xlabel('Theta (Polar Angle)')
    plt.ylabel('Density')
    plt.title('Theta (Polar Angle) Distribution')
    plt.grid(True)
    plt.savefig('ThetaDistribution.png')
    plt.close()

    # Plot phi (azimuthal angle) distribution
    plt.figure(figsize=(8, 6))
    plt.hist(phi, bins=50, density=True, alpha=0.7, color='orange')
    plt.xlabel('Phi (Azimuthal Angle)')
    plt.ylabel('Density')
    plt.title('Phi (Azimuthal Angle) Distribution')
    plt.grid(True)
    plt.savefig('PhiDistribution.png')
    plt.close()

    # Plot velocity magnitude distribution
    plt.figure(figsize=(8, 6))
    plt.hist(velocity_magnitudes, bins=50, density=True, alpha=0.7, color='purple')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Density')
    plt.title('Velocity Magnitude Distribution')
    plt.grid(True)
    plt.savefig('VelocityMagnitudeDistribution.png')
    plt.close()
    
    # Identify and print outlier velocities (beyond 3 sigma)
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    outliers = np.where(velocity_magnitudes > mean_velocity + 3 * std_velocity)[0]
    
    if len(outliers) > 0:
        print(f"Found {len(outliers)} outlier(s) in velocity data.")
        for idx in outliers[:10]:  # Show up to 10 outliers
            print(f"Outlier Velocity: {velocity_magnitudes[idx]:.2f}, Position: {positions[idx]}")

if __name__=="__main__":
    file_path = sys.argv[1]
    
    # Load data and run both analyses
    data = load_bods_file(file_path)
    analyze_halo_data(data)
    analyze_halo_data_density(data)
