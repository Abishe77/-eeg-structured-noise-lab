import numpy as np


def create_electrodes(radius, number_of_electrodes):
    electrodes = []

    z_spacing = (2 * radius) / number_of_electrodes

    for i in range(number_of_electrodes):
        z = radius - (i + 0.5) * z_spacing
        phi = np.arccos(z / radius)
        theta = i * np.deg2rad(137.5)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)

        electrodes.append([x, y, z])

    return np.array(electrodes)


def create_dipole(phi_degrees, theta_degrees, magnitude=1):
    phi = np.deg2rad(phi_degrees)
    theta = np.deg2rad(theta_degrees)

    px = magnitude * np.sin(phi) * np.cos(theta)
    py = magnitude * np.sin(phi) * np.sin(theta)
    pz = magnitude * np.cos(phi)

    return np.array([px, py, pz])


def create_candidates(radius, spacing=1):
    candidates = []

    for x in np.arange(-radius, radius, spacing):
        for y in np.arange(-radius, radius, spacing):
            for z in np.arange(-radius, radius, spacing):
                if x**2 + y**2 + z**2 < radius**2:
                    candidates.append([x, y, z])

    return np.array(candidates)