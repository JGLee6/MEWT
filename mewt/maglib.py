# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:41:13 2020

@author: John Greendeer Lee
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.special as sp

mu_0 = np.pi*4e-7
magC = 1e-7


def mag_u_array(magnet1, magnet2):
    r"""
    Compute the magnetic force of all magnet2 points on magnet1.

    .. math::
        U &= -m_a \cdot B \\
        U &= -m_a \cdot (\mu_0(3\hat{r}(\hat{r}\cdot m_b)-m_b)/4\pi r^3) \\
        U &=-\mu_0(3(m_a\cdot\hat{r})(m_b\cdot\hat{r})-(m_a\cdot m_b))/4\pi r^3

    Inputs
    ------
    magnet1 : ndarray
        numpy array row, 1x8, of form [mass, x, y, z, s, sx, sy, sz]
    magnet2 : ndarray
        numpy array, Nx8

    Returns
    -------
    force : ndarray
        numpy array, 1x3, of force on magnet1 by all magnet2 elements
    """
    if np.ndim(magnet2) == 1:
        # Which way does the force act
        rvec = magnet2[1:4]-magnet1[1:4]
        m1hat = magnet1[5:]
        m2hat = magnet2[5:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2))
        rhat = rvec/r
        # Compute dot products of dipoles and rhat
        m1m2 = np.dot(m1hat, m2hat)
        rm1 = np.dot(rhat, m1hat)
        rm2 = np.dot(rhat, m2hat)
        # Compute force
        fac = magC*magnet1[4]*magnet2[4]/r**3
        energy = fac*(3*rm1*rm2 - m1m2)
    else:
        # Which way does the force act
        rvec = magnet2[:, 1:4]-magnet1[1:4]
        m1hat = magnet1[5:]
        m2hat = magnet2[:, 5:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2, 1))
        rhat = rvec*np.array([1/r]).T
        # Compute dot products of dipoles and rhat
        m1m2 = np.dot(m1hat, m2hat.T)
        rm1 = np.dot(rhat, m1hat)
        rm2 = np.diag(np.dot(rhat, m2hat.T))
        # compute force
        fac = magC*magnet1[4]*magnet2[:, 4]/r**3
        energy = fac*(3*rm1*rm2 - m1m2)

    return np.sum(energy)


def point_matrix_magnets_energy(magnet1, magnet2):
    """
    Compute the magnetostatic potential energy between array1 and array2.

    Inputs
    ------
    magnet1 : ndarray
        Mx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]
    magnet2 : ndarray
        Nx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]

    Returns
    -------
    energy : float
        potential energy
    """
    energy = 0
    if np.ndim(magnet1) == 1:
        energy = mag_u_array(magnet1, magnet2)
    else:
        for k in range(len(magnet1)):
            energyK = mag_u_array(magnet1[k, :], magnet2)

            energy += energyK

    return energy


def mag_ft_array(magnet1, magnet2):
    r"""
    Compute the magnetostatic force and torque from magnet2 points on magnet1.

    .. math::
        F &= \nabla(m_a\cdot B) = -\nabla_a(m_a\cdot B) \\
        F &= -\frac{3\mu_0}{4\pi r^4}
        [\hat{r}(m_a\cdot m_b) + m_a(\hat{r}\cdot m_b) + m_b(\hat{r}\cdot m_a)
             -5\hat{r}(\hat{r}\cdot m_a)(\hat{r}\cdot m_b))]

    .. math::
        T &= m_a \times B \\
        T &= \mu_0[3(m_b\cdot\hat{r})(m_a\times\hat{r})-(m_a\times m_b)]
        /4\pi r^3

    Inputs
    ------
    magnet1 : ndarray
        Mx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]
    magnet2 : ndarray
        Nx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]

    Returns
    -------
    force : ndarray
        numpy array, 1x3, of force on magnet1 by all magnet2 elements
    """
    if np.ndim(magnet2) == 1:
        # Which way does the force act
        rvec = magnet2[1:4]-magnet1[1:4]
        m1hat = magnet1[5:]
        m2hat = magnet2[5:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2))
        rhat = rvec/r
        # Compute dot products of dipoles and rhat
        m1m2 = np.dot(m1hat, m2hat)
        rm1 = np.dot(rhat, m1hat)
        rm2 = np.dot(rhat, m2hat)
        # Compute force
        fac = magC*magnet1[4]*magnet2[4]/r**3
        force = -3*fac*(rhat*(m1m2 - 5*rm1*rm2)+m1hat*(rm2)+m2hat*(rm1))/r
        torque = fac*(3*rm2*np.cross(m1hat, rhat) - np.cross(m1hat, m2hat))
    else:
        # Which way does the force act
        rvec = magnet2[:, 1:4]-magnet1[1:4]
        m1hat = magnet1[5:]
        m2hat = magnet2[:, 5:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2, 1))
        rhat = rvec*np.array([1/r]).T
        # Compute dot products of dipoles and rhat
        m1m2 = np.dot(m1hat, m2hat.T)
        rm1 = np.dot(rhat, m1hat)
        rm2 = np.diag(np.dot(rhat, m2hat.T))
        # compute force
        fac = magC*magnet1[4]*magnet2[:, 4]/r**3
        facf = 3*fac/r
        f1 = rhat.T.dot(facf*(m1m2-5*rm1*rm2))
        f2 = np.sum(np.outer(m1hat, facf*rm2).T, 0)
        f3 = m2hat.T.dot(facf*rm1)
        force = -(f1+f2+f3)
        torque = np.cross(m1hat, rhat).T.dot(3*fac*rm2)
        torque -= np.cross(m1hat, m2hat).T.dot(fac)

    return force, torque


def point_matrix_magnets(magnet1, magnet2):
    """
    Compute the force and 3-axis torque about the origin on array1 by array2
    from a magnetostatic potential.

    Inputs
    ------
    magnet1 : ndarray
        Mx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]
    magnet2 : ndarray
        Nx8, of form [mass_i, x_i, y_i, z_i, s_i, sx_i, sy_i, sz_i]

    Returns
    -------
    force : ndarray
        1x3 numpy array [f_x, f_y, f_z]
    torque : ndarray
        1x3 numpy array, [T_x, T_y, T_z]
    """
    force = np.zeros(3)
    torque = np.zeros(3)
    torque2 = np.zeros(3)

    if np.ndim(magnet1) == 1:
        force, torque = mag_ft_array(magnet1, magnet2)
        torque2 = np.cross(magnet1[1:4], force)
    else:
        for k in range(len(magnet1)):
            forceK, torqueK = mag_ft_array(magnet1[k, :], magnet2)
            torqueK2 = np.cross(magnet1[k, 1:4], forceK)

            force += forceK
            torque += torqueK
            torque2 += torqueK2

    return force, torque, torque2


def translate_dipole_array(pointMagnet, transVec):
    """
    Translate point magnets by transVec (a three vector).

    Inputs
    ------
    pointMagnet : ndarray
        Mx4 array of form [mass_i, x_i, y_i, z_i, sx_i, sy_i, sz_i]
    transVec : ndarray
        1x3 array

    Returns
    -------
    transArray : ndarray
        Mx4 translated array
    """
    if np.ndim(pointMagnet) == 1:
        transArray = np.zeros(8)
        transArray[0] = pointMagnet[0]
        transArray[4:] = pointMagnet[4:]
        transArray[1:4] = pointMagnet[1:4]+transVec
    else:
        transArray = np.zeros([len(pointMagnet), 8])
        transArray[:, 0] = pointMagnet[:, 0]
        transArray[:, 1:4] = pointMagnet[:, 1:4]+transVec
        transArray[:, 4:] = pointMagnet[:, 4:]

    return transArray


def rotate_dipole_array(pointMagnet, theta, rotVec):
    """
    Rotate pointMass by angle (in radians) about vector from origin.

    Uses Rodrigues' Formula:
    http://mathworld.wolfram.com/RodriguesRotationFormula.html

    This function is different from the simple rotation for the gravitational
    point masses in that we must consider what happens to the dipole
    orientation as we rotate. We choose to also rotate the dipole, which is in
    some sense equivalent to an opposite rotation of the frame of reference.
    We should also find that this matches well with the rotation of multipole
    moments from magnetic solids.

    Inputs
    Returns
    """
    norm = np.sqrt(np.dot(rotVec, rotVec))
    unit = rotVec/norm

    W = np.array([[0, -unit[2], unit[1]],
                  [unit[2], 0, -unit[0]],
                  [-unit[1], unit[0], 0]])
    R = np.identity(3)+np.sin(theta)*W+2*(np.sin(theta/2.)**2)*W.dot(W)

    if np.ndim(pointMagnet) == 1:
        rotArray = np.zeros(8)
        rotArray[0] = pointMagnet[0]
        rotArray[4] = pointMagnet[4]
        rotArray[1:4] = R.dot(pointMagnet[1:4])
        rotArray[5:] = R.dot(pointMagnet[5:])
    else:
        rotArray = np.zeros([len(pointMagnet), 8])
        rotArray[:, 0] = pointMagnet[:, 0]
        rotArray[:, 4] = pointMagnet[:, 4]
        rotArray[:, 1:4] = np.dot(R, pointMagnet[:, 1:4].T).T
        rotArray[:, 5:8] = np.dot(R, pointMagnet[:, 5:8].T).T

    return rotArray


def display_dipoles(pm1, pm2, length=1):
    """
    Create a 3-dimensional plot of the two point-mass arrays pm1 and pm2.

    Inputs
    ------
    pm1 : ndarray
        N1x4 array of first set of point masses [m, x, y, z, s, sx, sy, sz]
    pm2 : ndarray
        N2x4 array of second set of point masses [m, x, y, z, s, sx, sy, sz]

    Returns
    -------
    fig : matplotlib.pyplot.figure object
        Figure object
    ax : matplotlib.pyplot.axes object
        Axes object
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.quiver(pm1[:, 1], pm1[:, 2], pm1[:, 3], pm1[:, 5], pm1[:, 6], pm1[:, 7],
              label='magnet1', alpha=.5, normalize=True, arrow_length_ratio=.8,
              length=length)
    ax.quiver(pm2[:, 1], pm2[:, 2], pm2[:, 3], pm2[:, 5], pm2[:, 6], pm2[:, 7],
              label='magnet2', alpha=.5, normalize=True, arrow_length_ratio=.8,
              color='C1', length=length)
    ax.legend()
    return fig, ax


def dmoment(l, m, magArray):
    """
    Computes the small q(l, m) inner multipole moment of a point dipole array.

    Evaluates the dot product with the derivative of the solid harmonic at
    each point-dipole position.

    Inputs
    ------
    l : int
        Multipole moment order
    m : int
        Multipole moment order, m < l
    magArray : ndarray
        Nx8 array of point masses [m, x, y, z, s, sx, sy, sz]

    Returns
    -------
    qlm : complex
        Complex-valued inner multipole moment
    """
    if l == 0:
        return 0
    r = np.sqrt(magArray[:, 1]**2 + magArray[:, 2]**2 + magArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.arccos(magArray[rids, 3]/r[rids])
    phi = np.arctan2(magArray[rids, 2], magArray[rids, 1]) % (2*np.pi)
    # Varshalovich p21, eqn37
    ep1 = -1/np.sqrt(2)*(magArray[:, 5] + 1j*magArray[:, 6])
    e0 = magArray[:, 7]
    em1 = 1/np.sqrt(2)*(magArray[:, 5] - 1j*magArray[:, 6])
    # then varshalovich p31 eqn4 (D = -ep1*Dm1 + e0*D0 - em1*Dp1)
    # Find contributions to each moment from each term in gradient
    # Now Varshalovich p160 eqn10, eqn11
    # Varshalovich p225 eqn17
    rl = r**(l-1)
    rlfac = rl*np.sqrt(l*(2*l+1)/(2*l*(2*l-1)))
    qlm0, qlmm1, qlmp1 = 0, 0, 0
    if abs(m) <= l-1:
        y0 = np.sqrt((l-m)*(l+m)*2)*np.conj(sp.sph_harm(m, l-1, phi, theta))
        qlm0 = e0*y0
    if abs(m-1) <= l-1:
        ym = np.sqrt((l+m)*(l+m-1))*np.conj(sp.sph_harm(m-1, l-1, phi, theta))
        qlmm1 = -em1*ym
    if abs(m+1) <= l-1:
        yp = np.sqrt((l-m)*(l-m-1))*np.conj(sp.sph_harm(m+1, l-1, phi, theta))
        qlmp1 = -ep1*yp
    # Now Varshalovich p160 eqn13
    qlm = np.sum(magArray[:, 4]*rlfac*(qlm0 + qlmm1 + qlmp1))
    return qlm


def dmoments(l, magArray):
    """
    Compute all small q(l, m) inner multipole moment of a point dipole array.

    Evaluates the dot product with the derivative of the solid harmonic at
    each point-dipole position.

    Inputs
    ------
    l : int
        Maximum multipole moment order
    magArray : ndarray
        Nx8 array of point masses [m, x, y, z, s, sx, sy, sz]

    Returns
    -------
    qlms : ndarry, complex
        Complex-valued inner multipole moments up to order l.
    """
    qlms = np.zeros([l+1, 2*l+1], dtype='complex')
    r = np.sqrt(magArray[:, 1]**2 + magArray[:, 2]**2 + magArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.zeros(len(magArray))
    phi = np.zeros(len(magArray))
    theta[rids] = np.arccos(magArray[rids, 3]/r[rids])
    phi[rids] = np.arctan2(magArray[rids, 2], magArray[rids, 1]) % (2*np.pi)
    ep1 = -1/np.sqrt(2)*(magArray[:, 5] + 1j*magArray[:, 6])
    e0 = magArray[:, 7]
    em1 = 1/np.sqrt(2)*(magArray[:, 5] - 1j*magArray[:, 6])
    # Never have a l=0 moment for dipoles
    for n in range(1, l+1):
        rl = r**(n-1)
        rlfac = rl*np.sqrt(n*(2*n+1)/(2*n*(2*n-1)))
        for m in range(n+1):
            qlm0, qlmm1, qlmp1 = 0, 0, 0
            if abs(m) <= n-1:
                y0 = np.sqrt((n-m)*(n+m)*2)*np.conj(sp.sph_harm(m, n-1, phi, theta))
                qlm0 = e0*y0
            if abs(m-1) <= n-1:
                ym = np.sqrt((n+m)*(n+m-1))*np.conj(sp.sph_harm(m-1, n-1, phi, theta))
                qlmm1 = -em1*ym
            if abs(m+1) <= n-1:
                yp = np.sqrt((n-m)*(n-m-1))*np.conj(sp.sph_harm(m+1, n-1, phi, theta))
                qlmp1 = -ep1*yp
            qlms[n, l+m] = np.sum(magArray[:, 4]*rlfac*(qlm0 + qlmm1 + qlmp1))

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-l, l+1)
    fac = (-1)**(np.abs(ms))
    qlms += np.conj(np.fliplr(qlms))*fac
    qlms[:, l] /= 2

    return qlms


def Dmomentsb(l, magArray):
    """
    Compute all big Q(l, m) outer multipole moment of a point dipole array.

    Evaluates the dot product with the derivative of the irregular solid
    harmonic at each point-dipole position.

    Inputs
    ------
    l : int
        Maximum multipole moment order
    magArray : ndarray
        Nx8 array of point masses [m, x, y, z, s, sx, sy, sz]

    Returns
    -------
    qlms : ndarry, complex
        Complex-valued inner multipole moments up to order l.
    """
    Qlmsb = np.zeros([l+1, 2*l+1], dtype='complex')
    r = np.sqrt(magArray[:, 1]**2 + magArray[:, 2]**2 + magArray[:, 3]**2)
    if (r == 0).any():
        print('Outer multipole moments cannot be evaluated at the origin.')
        return Qlmsb
    theta = np.arccos(magArray[:, 3]/r)
    phi = np.arctan2(magArray[:, 2], magArray[:, 1]) % (2*np.pi)
    ep1 = -1/np.sqrt(2)*(magArray[:, 5] + 1j*magArray[:, 6])
    e0 = magArray[:, 7]
    em1 = 1/np.sqrt(2)*(magArray[:, 5] - 1j*magArray[:, 6])
    for n in range(l+1):
        rl = r**(-n-2)
        rlfac = rl*np.sqrt((n+1)*(2*n+1)/(2*(n+1)*(2*n+3)))
        for m in range(n+1):
            Qlm0, Qlmm1, Qlmp1 = 0, 0, 0
            if abs(m) <= n+1:
                y0 = -np.sqrt((n-m+1)*(n+m+1)*2)*sp.sph_harm(m, n+1, phi, theta)
                Qlm0 = e0*y0
            if abs(m+1) <= n+1:
                yp1 = -np.sqrt((n+m+1)*(n+m+2))*sp.sph_harm(m+1, n+1, phi, theta)
                Qlmm1 = -em1*yp1
            if abs(m-1) <= n+1:
                ym1 = -np.sqrt((n-m+1)*(n-m+2))*sp.sph_harm(m-1, n+1, phi, theta)
                Qlmp1 = -ep1*ym1
            Qlmsb[n, l+m] = np.sum(magArray[:, 4]*rlfac*(Qlm0 + Qlmm1 + Qlmp1))

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-l, l+1)
    fac = (-1)**(np.abs(ms))
    Qlmsb += np.conj(np.fliplr(Qlmsb))*fac
    Qlmsb[:, l] /= 2

    return Qlmsb
