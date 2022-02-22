# MEWT (Magnetic E&ouml;t-Wash Toolkit)
## A package for calculating magnetostatic forces and torques from the E&ouml;t-Wash group
## John Greendeer Lee
[![Build Status](https://app.travis-ci.com/JGLee6/MEWT.svg?token=uUPz8q5eKZz3VguLVx33&branch=main)](https://app.travis-ci.com/JGLee6/MEWT)


We present MEWT: the Magnetic E&ouml;t-Wash Toolkit. This toolkit uses two methods to calculate magnetostatic interactions between objects with a fixed magnetization. The first method works by discretizing any shape into a 3-dimensional array of point-dipoles. We may then easily calculate forces and torques by summing over every possible pair of point interactions. Our second method uses known multipole moments of various primitive shapes and combines them with techniques for calculating translations, rotations, torques, and forces between moments similar to [NEWT](https://github.com/4kbt/NewtonianEotWashToolkit) (our Newtonian gravity toolkit). We have paired the simplicity of point-dipoles with the fast and accurate multipole calculations, with plenty of comparisons between the two.

## Point Dipoles
A point dipole array is an Nx8 array of points with [m, x, y, z, s, sx, sy, sz] specifying the mass and position as well as the dipole moment and orientation.

## Multipoles

### To Do
- [ ] minus sign torque \(NEWT\)
- [ ] phi_0 for x,y shapes
- [ ] visualization
- [ ] point-dipole as strictly amplitude and direction
- [ ] tetrahedron moments