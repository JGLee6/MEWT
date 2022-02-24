#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:29:52 2022

@author: John Greendeer Lee
"""
import numpy as np
import matplotlib.pyplot as plt
import mewt.maglib as mglb
import mewt.maglibShapes as mshp
import newt.multipoleLib as mplb

# Create a cylinder
cyl = mshp.annulus(1, 0, 1, 1, 1, 1, 0, 0, 2, 2)
# Inner cylinders on radius of 1m
cyl1 = mglb.translate_dipole_array(cyl, [2, 0, 0])
# Outer cylinders on radius of 5m
cyl2 = mglb.translate_dipole_array(cyl, [6, 0, 0])
# Combination of three inner cylinders
m1 = np.concatenate([cyl1, mglb.rotate_dipole_array(cyl1, 2*np.pi/3, [0, 0, 1]),
                     mglb.rotate_dipole_array(cyl1, -2*np.pi/3, [0, 0, 1])])
# Combination of three outer cylinders
m2 = np.concatenate([cyl2, mglb.rotate_dipole_array(cyl2, 2*np.pi/3, [0, 0, 1]),
                     mglb.rotate_dipole_array(cyl2, -2*np.pi/3, [0, 0, 1])])
fig, ax = mglb.display_dipoles(m1, m2, length=2)
ax.set_zlim([-6, 6])

# Calculate torque about z-axis as function of angle of m1
angles = np.arange(360)*np.pi/180
torques_z = np.zeros(360)
for k, angle in enumerate(angles):
    m1b = mglb.rotate_dipole_array(m1, angle, [0, 0, 1])
    _, torq, torq2 = mglb.point_matrix_magnets(m1b, m2)
    torques_z[k] = torq[2]+torq2[2]

fig, ax = plt.subplots(1, 1)
ax.plot(angles, torques_z)
ax.set_xlabel("angle [rad]")
ax.set_ylabel("torque [Nm]")

# Calculate torque by multipole method
lmax = 10
dlm = mglb.dmoments(lmax, m1)
Dlm = mglb.Dmomentsb(lmax, m2)
mplb.torque_lm(lmax, dlm, Dlm)
tlm, tc, ts = mplb.torque_lm(lmax, dlm, Dlm)
ts *= mglb.magC/mplb.BIG_G

# the 3-omega torque is
print(ts[2])
fig, ax = plt.subplots(1, 1)
for k in range(1, 10):
    ax.plot(angles, np.imag(ts[k])*np.sin(k*angles), label=str(k)+r'$\omega$')
ax.set_xlabel("angle [rad]")
ax.legend()
ax.set_ylabel("torque [Nm]")
