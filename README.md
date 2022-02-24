# MEWT (Magnetic E&ouml;t-Wash Toolkit)
## A package for calculating magnetostatic forces and torques from the E&ouml;t-Wash group
## John Greendeer Lee
[![Build Status](https://app.travis-ci.com/JGLee6/MEWT.svg?token=uUPz8q5eKZz3VguLVx33&branch=main)](https://app.travis-ci.com/JGLee6/MEWT)


We present MEWT: the Magnetic E&ouml;t-Wash Toolkit. This toolkit uses two methods to calculate magnetostatic interactions between objects with a fixed magnetization. The first method works by discretizing any shape into a 3-dimensional array of point-dipoles. We may then easily calculate forces and torques by summing over every possible pair of point interactions. Our second method uses known multipole moments of various primitive shapes and combines them with techniques for calculating translations, rotations, torques, and forces between moments similar to [NEWT](https://github.com/4kbt/NewtonianEotWashToolkit) (our Newtonian gravity toolkit). We have paired the simplicity of point-dipoles with the fast and accurate multipole calculations, with plenty of comparisons between the two.

## Point Dipoles
A point dipole array is an Nx8 array of points with [m, x, y, z, s, sx, sy, sz] specifying the mass and position as well as the dipole moment and orientation.

Here's an example, rendering two pairs of 3 cylinders \(approximated by 4 point-dipoles\) with magnetization oriented to the angle of the cylinder position.

```python
import numpy as np
import mewt.maglib as mglb
import mewt.maglibShapes as mshp

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
```
![Example: Three cylinders approximated by 4 point-dipoles comprising object 1(blue), and object 2(orange)](/examples/maglibEx1.png)

And we can also calculate the torque about the z-axis at different angles of the inner magnets via point-to-point interactions:
```python
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
```
![Example: Calculating dipole-dipole torques from 3-fold cylinders)](/examples/maglibEx2.png)


Or alternatively, we can find the multipole moments of the point-dipoles and calculate the torque about the z-axis directly.
```python
import newt.multipoleLib as mplb

lmax = 10
dlm = mglb.dmoments(lmax, m1)
Dlm = mglb.Dmomentsb(lmax, m2)
mplb.torque_lm(lmax, dlm, Dlm)
tlm, tc, ts = mplb.torque_lm(lmax, dlm, Dlm)
ts *= mglb.magC/mplb.BIG_G

# Plot the individual sinusoid contributions
fig, ax = plt.subplots(1, 1)
for k in range(1, 10):
    ax.plot(angles, np.imag(ts[k])*np.sin(k*angles), label=str(k)+r'$\omega$')
ax.set_xlabel("angle [rad]")
ax.legend()
ax.set_ylabel("torque [Nm]")
```
![Example: Calculating dipole-dipole torques from 3-fold cylinders via multipole method)](/examples/maglibEx3.png)

## Multipoles

### To Do
- [ ] minus sign torque \(NEWT\)
- [ ] phi_0 for x,y shapes
- [ ] visualization
- [ ] point-dipole as strictly amplitude and direction
- [ ] tetrahedron moments