#!/usr/bin/env python3

'''
UNIT CONVERSION

    Converts SI units to simulation units.

USAGE:
    python convert.py [value] [SI unit]

    python convert.py 1 m/s
    python convert.py 1 velocity
    python convert.py 1 v
'''

import argparse
import astropy.constants as const
import astropy.units as unit

# Simulation units
Tsim = 1e9 * unit.yr                 # Time unit: 1 gigayear
Lsim = 1 * unit.kpc                  # Length unit: 1 kiloparsec
Msim = 222288.47 * const.M_sun    # Mass unit: 222,288.47 solar masses

# Gravitational constant in simulation units
G_standard = const.G
G_sim = G_standard.to(Lsim**3 / (Msim * Tsim**2))
assert abs(G_sim.value - 1) < 1e-3, "G in simulation units should be 1."

# Conversion functions
def kg_to_sim(mass_kg):
    return (mass_kg * unit.kg).to(Msim).value, "simulation masses"

def sm_to_sim(mass_sm):
    return (mass_sm * const.M_sun).to(Msim).value, "simulation masses"

def m_to_sim(length_m):
    return (length_m * unit.m).to(Lsim).value, "kiloparsecs"

def s_to_sim(time_s):
    return (time_s * unit.s).to(Tsim).value, "gigayears"

def v_to_sim(velocity_mps):
    return (velocity_mps * unit.m / unit.s).to(Lsim / Tsim).value, "kpc / gyr"

def F_to_sim(force_N):
    return (force_N * unit.N).to(Msim * Lsim / Tsim**2).value, "M_sim * kpc / gyr^2"

def rho_to_sim(density_kgm3):
    return (density_kgm3 * unit.kg / unit.m**3).to(Msim / Lsim**3).value, "M_sim / kpc^3"

def p_to_sim(momentum_kgmps):
    return (momentum_kgmps * (unit.kg * unit.m / unit.s)).to(Msim / Lsim**3).value, "M_sim * kpc / gyr^2"

def ly_to_sim(length_ly):
    return (length_ly * unit.lyr).to(Lsim).value, "kpc"

CONVERSIONS = {
    "kg": kg_to_sim,
    "mass": kg_to_sim,
    "sm": sm_to_sim,
    "solar mass": sm_to_sim,
    "solar masses": sm_to_sim,
    "M_s": sm_to_sim,
    "m_s": sm_to_sim,
    "ms": sm_to_sim,
    "m": m_to_sim,
    "length": m_to_sim,
    "distance": m_to_sim,
    "s": s_to_sim,
    "t": s_to_sim,
    "time": s_to_sim,
    "v": v_to_sim,
    "m/s": v_to_sim,
    "velocity": v_to_sim,
    "F": F_to_sim,
    "f": F_to_sim,
    "kgm/s^2": F_to_sim,
    "kg m/s^2": F_to_sim,
    "force": F_to_sim,
    "kg/m^3": rho_to_sim,
    "rho": rho_to_sim,
    "density": rho_to_sim,
    "p": p_to_sim,
    "mv": p_to_sim,
    "kgm/s": p_to_sim,
    "kg m/s": p_to_sim,
    "momentum": p_to_sim,
    "lyr": ly_to_sim,
    "ly": ly_to_sim,
    "lightyear": ly_to_sim
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SI units to simulation units.")
    parser.add_argument("value", type=float, help="The numeric value to convert.")
    parser.add_argument("unit", choices=CONVERSIONS.keys(), help="The type of unit to convert (mass, length, time, velocity, force, density).")
    args = parser.parse_args()

    conversion_function = CONVERSIONS[args.unit]
    result, units = conversion_function(args.value)
    print(f"{args.value} {args.unit} in simulation units: {result:.6e} {units}")
