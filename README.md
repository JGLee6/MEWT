# MEWT (Magnetic E&ouml;t-Wash Toolkit)
## A package for calculating magnetostatic forces and torques from the E&ouml;t-Wash group
## John Greendeer Lee


We present MEWT: the Magnetic E&ouml;t-Wash Toolkit. This toolkit uses two methods to calculate magnetostatic interactions between objects with a fixed magnetization. The first method works by discretizing any shape into a 3-dimensional array of point-dipoles. We may then easily calculate forces and torques by summing over every possible pair of point interactions. Our second method uses known multipole moments of various shapes similar to NEWT (our Newtonian gravity toolkit) and pairs them with techniques for calculating torques and forces between moments. We have paired the simple and easily-comprehended method of point-dipoles with the fast and powerful method of multipole moments with plenty of comparisons between the two.

## Point Dipoles
A point dipole array is an Nx8 array of points with [m, x, y, z, s, sx, sy, sz] specifying the mass and position as well as the dipole moment and orientation.

### To Do
- [X] Test Rect
	- [X] x
	- [X] y
	- [X] z
- [X] Test Tri
	- [X] x
	- [X] y
	- [X] z
- [X] Test Cyl
	- [X] x
	- [X] y
	- [X] z
	- [X] rho
	- [X] rho, beta != pi
	- [X] phi
	- [X] phi, beta != pi
- [X] Test Cone
	- [X] x
	- [X] y
	- [X] z
	- [X] rho
	- [X] rho, beta != pi
	- [X] phi
	- [ ] phi, beta != pi, -> minus sign
- [ ] fix y-axis error in moments from point dipole
- [X] point mags