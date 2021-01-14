# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:35:15 2020

@author: jgl6
"""
import numpy as np
import maglib as mglb
import maglibShapes as mshp
import newt.multipoleLib as mplb
import newt.translations as trs
import newt.rotations as rot
import mqlm


def test_dmomz():
    """
    Compares the inner moment of dipole at the origin with vertical orientation
    to the a priori calculation.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    pred = np.zeros([lmax+1, 2*lmax+1], dtype='complex')
    pred[1, lmax] = np.sqrt(3/(4*np.pi))
    assert (np.abs(dmom - pred) < 10*np.finfo(float).eps).all()
    # Check if moved to [1, 0, 0]
    mag1 = np.array([[1, 1, 0, 0, 1, 0, 0, 1]])
    dmom = mglb.dmoments(lmax, mag1)
    predx = trs.translate_qlm(pred, [1, 0, 0])
    assert (np.abs(dmom - predx) < 50*np.finfo(float).eps).all()
    # Check if moved to [0, 1, 0]
    mag1 = np.array([[1, 0, 1, 0, 1, 0, 0, 1]])
    dmom = mglb.dmoments(lmax, mag1)
    predy = trs.translate_qlm(pred, [0, 1, 0])
    assert (np.abs(dmom - predy) < 50*np.finfo(float).eps).all()
    # Check if moved to [0, 0, 1]
    mag1 = np.array([[1, 0, 0, 1, 1, 0, 0, 1]])
    dmom = mglb.dmoments(lmax, mag1)
    predz = trs.translate_qlm(pred, [0, 0, 1])
    assert (np.abs(dmom - predz) < 150*np.finfo(float).eps).all()


def test_dmomx():
    """
    Compares the inner moment of dipole at the origin with horizontal x-
    orientation to the a priori calculation.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 1, 0, 0]])
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    pred = np.zeros([lmax+1, 2*lmax+1], dtype='complex')
    pred[1, lmax+1] = -np.sqrt(3/(4*np.pi))/np.sqrt(2)
    pred[1, lmax-1] = np.sqrt(3/(4*np.pi))/np.sqrt(2)
    assert (np.abs(dmom - pred) < 10*np.finfo(float).eps).all()
    # Check if moved to [1, 0, 0]
    mag1 = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predx = trs.translate_qlm(pred, [1, 0, 0])
    assert (np.abs(dmom - predx) < 150*np.finfo(float).eps).all()
    # Check if moved to [0, 1, 0]
    mag1 = np.array([[1, 0, 1, 0, 1, 1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predy = trs.translate_qlm(pred, [0, 1, 0])
    assert (np.abs(dmom - predy) < 150*np.finfo(float).eps).all()
    # Check if moved to [0, 0, 1]
    mag1 = np.array([[1, 0, 0, 1, 1, 1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predz = trs.translate_qlm(pred, [0, 0, 1])
    assert (np.abs(dmom - predz) < 200*np.finfo(float).eps).all()


def test_dmomy():
    """
    Compares the inner moment of dipole at the origin with horizonatl y-
    orientation to the a priori calculation.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 1, 0]])
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    pred = np.zeros([lmax+1, 2*lmax+1], dtype='complex')
    pred[1, lmax+1] = np.sqrt(3/(4*np.pi))/np.sqrt(2)*1j
    pred[1, lmax-1] = np.sqrt(3/(4*np.pi))/np.sqrt(2)*1j
    assert (np.abs(dmom - pred) < 10*np.finfo(float).eps).all()
    # Check if moved to [1, 0, 0]
    mag1 = np.array([[1, 1, 0, 0, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predx = trs.translate_qlm(pred, [1, 0, 0])
    assert (np.abs(dmom - predx) < 150*np.finfo(float).eps).all()
    # Check if moved to [0, 1, 0]
    mag1 = np.array([[1, 0, 1, 0, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predy = trs.translate_qlm(pred, [0, 1, 0])
    assert (np.abs(dmom - predy) < 150*np.finfo(float).eps).all()
    # Check if moved to [0, 0, 1]
    mag1 = np.array([[1, 0, 0, 1, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag1)
    predz = trs.translate_qlm(pred, [0, 0, 1])
    assert (np.abs(dmom - predz) < 200*np.finfo(float).eps).all()


def test_rot():
    lmax = 10
    # z-dip at origin -> mom -> [0, 0, 1]
    magz = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    dmomz = mglb.dmoments(lmax, magz)
    dmomzt = trs.translate_qlm(dmomz, [0, 0, 1])
    # z-dip at [0, 0, 1] -> mom
    magtz = np.array([[1, 0, 0, 1, 1, 0, 0, 1]])
    dmomtz = mglb.dmoments(lmax, magtz)
    assert (np.abs(dmomtz - dmomzt) < 2e2*np.finfo(float).eps).all()
    # x-dip at origin -> mom -> [1, 0, 0]
    magx = np.array([[1, 0, 0, 0, 1, 1, 0, 0]])
    dmomx = mglb.dmoments(lmax, magx)
    dmomxt = trs.translate_qlm(dmomx, [1, 0, 0])
    # x-dip at [1, 0, 0] -> mom
    magtx = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    dmomtx = mglb.dmoments(lmax, magtx)
    assert (np.abs(dmomtx - dmomxt) < 2e2*np.finfo(float).eps).all()
    # z-dip at [0, 0, 1] -> rotate z to x-axis
    dmomxrt = rot.rotate_qlm(dmomzt, 0, np.pi/2, 0)
    assert (np.abs(dmomxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
    # np.where(np.abs(dmomxrt - dmomxt) > 2e2*np.finfo(float).eps)


def test_ft_a():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis both with
    vertical unit dipole moments.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    mag2 = np.array([[1, 1, 0, 0, 1, 0, 0, 1]])
    fpred = np.array([3*mglb.mu_0/(4*np.pi), 0, 0])
    tpred = np.array([0, 0, 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    force = -np.real(force)*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    # assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    # assert (np.abs(tpred - torque) < 10*np.finfo(float).eps).all()


def test_ft_b():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis, one at
    the origin with vertical unit dipole moment and the other with x-oriented
    unit dipole moment.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    mag2 = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    fpred = np.array([0, 0, 3*mglb.mu_0/(4*np.pi)])
    tpred = np.array([0, mglb.mu_0/(4*np.pi), 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t2 - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    force = -np.real(force)*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    # assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    # assert (np.abs(tpred - torque) < 10*np.finfo(float).eps).all()


def test_ft_c():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis, both with
    x-oriented unit dipole moments.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 1, 0, 0]])
    mag2 = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    fpred = np.array([-6*mglb.mu_0/(4*np.pi), 0, 0])
    tpred = np.array([0, 0, 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    force = -np.real(force)*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    # assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    # assert (np.abs(tpred - torque) < 10*np.finfo(float).eps).all()


def test_Dmombs():
    """
    Creates a pair of point dipole at [1, 0, 0] and [-1, 0, 0] with vertically
    oriented moments. Then compares the outer moments of the vertically
    translated ([0, 0, 5]) points computed in two ways. The first method is
    through translating the points and computing the outer moments of point
    dipoles. The second method, computes the inner moments of point dipoles and
    translates to outer moments using a translation method of inner-outer
    moments.
    """
    mag2 = np.array([[1, 1, 0, 0, 1, 0, 0, 1], [1, -1, 0, 0, 1, 0, 0, 1]])
    lmax = 20
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, [0, 0, 5])
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, [0, 0, 5])
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()


def test_force():
    """
    """
    thetak = np.pi/6
    mag3p = np.array([[1, 1, 0, .25, 1, 1, 0, 0],
                      [1, np.cos(2*np.pi/3), np.sin(2*np.pi/3), .25,
                       1, 1, 0, 0],
                      [1, np.cos(2*np.pi/3), -np.sin(2*np.pi/3), .25,
                       1, 1, 0, 0]])
    mag3a = np.array([[1, np.cos(thetak), np.sin(thetak), -.25, 1, 1, 0, 0],
                      [1, np.cos(thetak+2*np.pi/3), np.sin(thetak+2*np.pi/3),
                       -.25, 1, 1, 0, 0],
                      [1, np.cos(thetak-2*np.pi/3), np.sin(thetak-2*np.pi/3),
                       -.25, 1, 1, 0, 0]])
    frc, trq2, trq = mglb.point_matrix_magnets(mag3p, mag3a)
    frcb, trqb = 0, 0
    for k in range(len(mag3p)):
        for m in range(len(mag3a)):
            frckm, trqkm = mglb.mag_ft_array(mag3p[k], mag3a[m])
            frcb += frckm
            trqb += trqkm
    assert (np.abs(frc - frcb) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq2 - trqb) < 10*np.finfo(float).eps).all()


def energy_3om(theta, z):
    """
    Assumes triplets of only x-oriented dipoles at radius of 1m with separation
    z. The three-fold attractor dipoles are rotated by an angle theta. Returns
    the energy
    """
    U = 0
    for k in range(3):
        for m in range(3):
            rvec = np.array([np.cos(2*np.pi*k/3)-np.cos(2*np.pi*m/3+theta),
                             np.sin(2*np.pi*k/3)-np.sin(2*np.pi*m/3+theta), z])
            r = np.sqrt(np.sum(rvec**2))
            fackm = 3*rvec[0]**2/r**2 - 1
            U += fackm/r**3
    U *= mglb.mu_0/(4*np.pi)
    return U


def energy_3omb(theta, z):
    """
    Assumes triplets of only x-oriented dipoles at radius of 1m with separation
    z. The three-fold attractor dipoles are rotated by an angle theta. Returns
    the energy
    """
    U = 0
    for k in range(3):
        for m in range(3):
            fackm = 2 + np.cos(4*np.pi*k/3) - 2*np.cos(2*np.pi/3*(k-m)-theta)
            fackm -= 2*np.cos(2*np.pi/3*(k+m)+theta)
            fackm += np.cos(4*np.pi*m/3 + 2*theta)
            rkm = np.sqrt(2+z**2-2*np.cos(2*np.pi/3*(k-m)-theta))
            U += (3/2*fackm/rkm**2 - 1)/rkm**3
    U *= mglb.mu_0/(4*np.pi)
    return U


# Energy with 3-fold stuff
theta = np.arange(360)*np.pi/180
u3om = energy_3om(theta, .5)
mag3p = np.array([[1, 1, 0, .25, 1, 1, 0, 0],
                  [1, np.cos(2*np.pi/3), np.sin(2*np.pi/3), .25, 1, 1, 0, 0],
                  [1, np.cos(2*np.pi/3), -np.sin(2*np.pi/3), .25, 1, 1, 0, 0]])
u3ompd = np.zeros(len(theta))
for k in range(len(theta)):
    thetak = theta[k]
    mag3a = np.array([[1, np.cos(thetak), np.sin(thetak), -.25, 1, 1, 0, 0],
                      [1, np.cos(thetak+2*np.pi/3), np.sin(thetak+2*np.pi/3), -.25, 1, 1, 0, 0],
                      [1, np.cos(thetak-2*np.pi/3), np.sin(thetak-2*np.pi/3), -.25, 1, 1, 0, 0]])
    u3ompd[k] = mglb.point_matrix_magnets_energy(mag3p, mag3a)


# Check 18-fold points x vs z
def zpoint(theta, N, r=1):
    mags = np.zeros([N, 8])
    mags[:, 0] = 1
    mags[:, 4] = 1
    mags[:, 7] = 1
    dtheta = 2*np.pi/N
    for k in range(N):
        mags[k, 1] = r*np.cos(theta + k*dtheta)
        mags[k, 2] = r*np.sin(theta + k*dtheta)
    return mags


def xpoint(theta, N, r=1):
    """
    Makes N-fold axisymmetric pattern of x-oriented spins.
    """
    mags = np.zeros([N, 8])
    mags[:, 0] = 1
    mags[:, 4] = 1
    mags[:, 5] = 1
    dtheta = 2*np.pi/N
    for k in range(N):
        mags[k, 1] = r*np.cos(theta + k*dtheta)
        mags[k, 2] = r*np.sin(theta + k*dtheta)
    return mags


def ypoint(theta, N, r=1):
    """
    Makes N-fold axisymmetric pattern of x-oriented spins.
    """
    mags = np.zeros([N, 8])
    mags[:, 0] = 1
    mags[:, 4] = 1
    mags[:, 6] = 1
    dtheta = 2*np.pi/N
    for k in range(N):
        mags[k, 1] = r*np.cos(theta + k*dtheta)
        mags[k, 2] = r*np.sin(theta + k*dtheta)
    return mags


z = .21
p18 = xpoint(0, 18, 3)
p18 = mglb.translate_dipole_array(p18, [0, 0, z/2])
u18x = np.zeros(len(theta))
for k in range(len(theta)):
    thetak = theta[k]
    a18 = xpoint(thetak, 18, 2)
    a18 = mglb.translate_dipole_array(a18, [0, 0, -z/2])
    u18x[k] = mglb.point_matrix_magnets_energy(p18, a18)

p18 = zpoint(0, 18, 3)
p18 = mglb.translate_dipole_array(p18, [0, 0, z/2])
u18z = np.zeros(len(theta))
for k in range(len(theta)):
    thetak = theta[k]
    a18 = zpoint(thetak, 18, 2)
    a18 = mglb.translate_dipole_array(a18, [0, 0, -z/2])
    u18z[k] = mglb.point_matrix_magnets_energy(p18, a18)

p18 = ypoint(0, 18, 3)
p18 = mglb.translate_dipole_array(p18, [0, 0, z/2])
u18y = np.zeros(len(theta))
for k in range(len(theta)):
    thetak = theta[k]
    a18 = ypoint(thetak, 18, 2)
    a18 = mglb.translate_dipole_array(a18, [0, 0, -z/2])
    u18y[k] = mglb.point_matrix_magnets_energy(p18, a18)

# Big G UW
mupyrex = -13.52e-6
muss = .01
vrect = 76e-3*1.506e-3*41.6e-3
Bz = 60e-6
mu0 = 4e-7*np.pi
magrect = mupyrex*Bz/mu0*vrect
rsph = 124.89e-3
magsph = muss*Bz/mu0*(4/3)*np.pi*rsph**3
mrect = mshp.rectangle(magrect, 76e-3, 1.506e-3, 41.6e-3, mupyrex*Bz/mu0, 0, 0,
                       1, 20, 2, 10)
mag2 = np.array([[magsph, 16.76e-2, 0, 0, magsph, 0, 0, 1],
                 [magsph, -16.76e-2, 0, 0, magsph, 0, 0, 1]])
mag3 = mglb.translate_dipole_array(mag2, [0, 0, np.sqrt(2/3)*16.76e-2])
mag3 = np.concatenate([mag3,
                       mglb.translate_dipole_array(mag2, [0, 0, -np.sqrt(2/3)*16.76e-2])])
drect = mglb.dmoments(10, mrect)
Dmag3 = mglb.Dmomentsb(10, mag3)
tlm, tc, ts = mplb.torque_lm(10, drect, Dmag3)
ts*.5e-7/mplb.BIG_G*2

# BIPM
mucu = -9.63e-6
vcylt = np.pi*(55e-3/2)**2*55e-3
magcylt = mucu*Bz/mu0*vcylt
tcyl = mshp.annulus(magcylt, 0, 55e-3/2, 55e-3, mucu*Bz/mu0, 0, 0, 1, 10, 10)
rt = 120e-3
tcyl = mglb.translate_dipole_array(tcyl, [rt, 0, 0])

testpend = np.concatenate([tcyl,
                           mglb.rotate_dipole_array(tcyl, np.pi/2, [0, 0, 1]),
                           mglb.rotate_dipole_array(tcyl, -np.pi/2, [0, 0, 1]),
                           mglb.rotate_dipole_array(tcyl, np.pi, [0, 0, 1])])
vcyls = np.pi*(120e-3/2)**2*115e-3
magcyls = mucu*Bz/mu0*vcyls
scyl = mshp.annulus(magcyls, 0, 120e-3/2, 115e-3, mucu*Bz/mu0, 0, 0, 1, 10, 10)
rs = 214e-3
scyl = mglb.translate_dipole_array(scyl, [rs, 0, 0])
src_mass = np.concatenate([scyl,
                           mglb.rotate_dipole_array(scyl, np.pi/2, [0, 0, 1]),
                           mglb.rotate_dipole_array(scyl, -np.pi/2, [0, 0, 1]),
                           mglb.rotate_dipole_array(scyl, np.pi, [0, 0, 1])])
dtp = mglb.dmoments(10, testpend)
Dsm = mglb.Dmomentsb(10, src_mass)
tlmb, tcb, tsb = mplb.torque_lm(10, dtp, Dsm)


def test_rect():
    """
    Test z-oriented analytic magnetic multipole inner moments.
    """
    rqlm0 = mqlm.rect_prism_z(10, 1, 2, 3, 5, 0)
    rqlm1 = mqlm.rect_prism_z2(10, 1, 2, 3, 5, 0)
    rect2 = mshp.rectangle(1, 3, 5, 2, 1, 0, 0, 1, 10, 10, 10)
    rqlm2 = mglb.dmoments(10, rect2)
    tri0 = mqlm.tri_prism_z(10, 1, 2, 3/2, -2.5, 2.5)
    tri2 = mqlm.tri_prism_z(10, 1, 2, 5/2, -1.5, 1.5)
    tri2 = rot.rotate_qlm(tri2, 0, 0, np.pi/2)
    tri1 = rot.rotate_qlm(tri0, 0, 0, np.pi)
    tri3 = rot.rotate_qlm(tri2, 0, 0, np.pi)
    rqlm3 = tri0 + tri1 + tri2 + tri3
    assert (np.abs(rqlm0 - rqlm1) < 3e5*np.finfo(float).eps).all()
    assert (np.abs(rqlm0 - rqlm2) < 3e5*np.finfo(float).eps)[:3].all()
    assert (np.abs(rqlm0 - rqlm3) < 3e5*np.finfo(float).eps).all()


def test_recx():
    rqlm0 = mqlm.rect_prism_z(10, 1, 2, 3, 5, 0)
    rqlmx = mqlm.rect_prism_x(10, 1, 3, 2, 5, 0)
    rqlmx2 = mqlm.rect_prism_x2(10, 1, 3, 2, 5, 0)
    # rotate z-oriented to x-oriented
    rqlmx3 = rot.rotate_qlm(rqlm0, 0, np.pi/2, 0)
    tri0 = mqlm.tri_prism_x(10, 1, 3, 2/2, -2.5, 2.5)
    tri1 = mqlm.tri_prism_x(10, -1, 3, 2/2, -2.5, 2.5)
    tri1 = rot.rotate_qlm(tri1, 0, 0, np.pi)
    tri2 = mqlm.tri_prism_y(10, 1, 3, 5/2, -1, 1)
    tri3 = mqlm.tri_prism_y(10, -1, 3, 5/2, -1, 1)
    tri2 = rot.rotate_qlm(tri2, 0, 0, -np.pi/2)
    tri3 = rot.rotate_qlm(tri3, 0, 0, np.pi/2)
    rqlmx4 = tri0 + tri1 + tri2 + tri3
    rect5 = mshp.rectangle(1, 2, 5, 3, 1, 1, 0, 0, 10, 10, 10)
    rqlmx5 = mglb.dmoments(10, rect5)
    rect6 = mshp.rectangle(1, 2, 5, 3, 1, 1, 0, 0, 20, 20, 20)
    rqlmx6 = mglb.dmoments(10, rect6)
    assert (np.abs(rqlmx2 - rqlmx3) < 3e5*np.finfo(float).eps).all()
    assert (np.abs(rqlmx2 - rqlmx4) < 3e5*np.finfo(float).eps).all()
    # PointDipole gets bad after l=2
    assert (np.abs(rqlmx2 - rqlmx6) < 3e5*np.finfo(float).eps)[:3].all()


def test_recy():
    rqlm0 = mqlm.rect_prism_z(10, 1, 2, 3, 5, 0)
    rqlmy = mqlm.rect_prism_y(10, 1, 5, 3, 2, 0)
    rqlmy2 = mqlm.rect_prism_y2(10, 1, 5, 3, 2, 0)
    # rotate z-oriented to x-oriented
    rqlmy3 = rot.rotate_qlm(rqlm0, np.pi/2, np.pi/2, -np.pi/2)
    tri0 = mqlm.tri_prism_y(10, 1, 5, 3/2, -1, 1)
    tri1 = mqlm.tri_prism_y(10, -1, 5, 3/2, -1, 1)
    tri1 = rot.rotate_qlm(tri1, 0, 0, np.pi)
    tri2 = mqlm.tri_prism_x(10, 1, 5, 2/2, -1.5, 1.5)
    tri3 = mqlm.tri_prism_x(10, -1, 5, 2/2, -1.5, 1.5)
    tri2 = rot.rotate_qlm(tri2, 0, 0, np.pi/2)
    tri3 = rot.rotate_qlm(tri3, 0, 0, -np.pi/2)
    rqlmy4 = tri0 + tri1 + tri2 + tri3
    rect5 = mshp.rectangle(1, 3, 2, 5, 1, 0, 1, 0, 10, 10, 10)
    rqlmy5 = mglb.dmoments(10, rect5)
    rect6 = mshp.rectangle(1, 3, 2, 5, 1, 0, 1, 0, 20, 20, 20)
    rqlmy6 = mglb.dmoments(10, rect6)
    assert (np.abs(rqlmy2 - rqlmy3) < 3e5*np.finfo(float).eps).all()
    assert (np.abs(rqlmy4 - rqlmy3) < 3e5*np.finfo(float).eps).all()


def test_tri():
    # Make analytic versions
    H = 3
    dy = 5/2
    y1y = -1
    y2y = 1
    y1x = -2.5
    y2x = 2.5
    dx = 1
    L = 10
    tri1 = mqlm.tri_prism_x(L, 1, H, dx, y1x, y2x)
    tri2 = mqlm.tri_prism_y(L, 1, H, dy, y1y, y2y)
    # Create point-dipole versions
    N = 40
    tri1b = mshp.tri_prism(1, dx, y1x, y2x, H, 1, 1, 0, 0, N, N, N)
    tri2b = mshp.tri_prism(1, dy, y1y, y2y, H, 1, 0, 1, 0, N, N, N)
    tqlmx = mglb.dmoments(L, tri1b)
    tqlmy = mglb.dmoments(L, tri2b)
    assert (np.abs(tri1 - tqlmx) < 0.2)[:4].all()
    assert (np.abs(tri2 - tqlmy) < 0.2)[:4].all()
    tri0 = mqlm.tri_prism_z(L, 1, H, dx, y1x, y2x)
    tri0b = mshp.tri_prism(1, dx, y1x, y2x, H, 1, 0, 0, 1, N, N, N)
    tqlmz = mglb.dmoments(L, tri0b)
    assert (np.abs(tri0 - tqlmz) < 0.25)[:4].all()


def test_ann():
    H = 3
    IR = 1.5
    OR = 2
    L = 10
    N = 40
    beta = np.pi
    annx = mqlm.annulus_x(L, 1, H, IR, OR, 0, np.pi)
    anny = mqlm.annulus_y(L, 1, H, IR, OR, 0, np.pi)
    annx2 = rot.rotate_qlm(anny, 0, 0, -np.pi/2)
    assert (np.abs(annx - annx2) < 3e4*np.finfo(float).eps).all()
    annz = mqlm.annulus_z(L, 1, H, IR, OR, 0, beta)
    sannz = mshp.annulus(1, IR, OR, H, 1, 0, 0, 1, N, N)
    mannz = mglb.dmoments(L, sannz)
    assert (np.abs(annz - mannz) < 0.2)[:3].all()
    annr = mqlm.annulus_r(L, 1, H, IR, OR, 0, beta)
    sannr = mshp.wedge_rho(1, IR, OR, H, beta, 1, N, N)
    mannr = mglb.dmoments(L, sannr)
    assert (np.abs(annr - mannr) < 0.1)[:3].all()


def test_cone():
    H = 3
    R = 2
    L = 10
    N = 30
    beta = np.pi
    conx = mqlm.cone_x(L, 1, H, R, 0, beta)
    cony = mqlm.cone_y(L, 1, H, R, 0, beta)
    conx2 = rot.rotate_qlm(cony, 0, 0, -np.pi/2)
    assert (np.abs(conx - conx2) < 3e3*np.finfo(float).eps).all()
    sconx = mshp.cone(1, R, H, beta, 1, 1, 0, 0, N, N)
    mconx = mglb.dmoments(L, sconx)
    assert (np.abs(conx - mconx) < 0.1)[:3].all()
    conz = mqlm.cone_z(L, 1, H, R, 0, beta)
    sconz = mshp.cone(1, R, H, beta, 1, 0, 0, 1, N, N)
    mconz = mglb.dmoments(L, sconz)
    assert (np.abs(conz - mconz) < 0.1)[:3].all()
    conr = mqlm.cone_r(L, 1, H, R, 0, beta)
    sconr = mshp.cone_rho(1, R, H, beta, 1, N, N)
    mconr = mglb.dmoments(L, sconr)
    assert (np.abs(conr - mconr) < 0.2)[:3].all()
    conp = mqlm.cone_p(L, 1, H, R, 0, beta)
    sconp = mshp.cone_phi(1, R, H, beta, 1, N, N)
    mconp = mglb.dmoments(L, sconp)
    assert (np.abs(conp - mconp) < 0.2)[:3].all()
    beta = np.pi/6
    conp = mqlm.cone_p(L, 1, H, R, 0, beta)
    sconp = mshp.cone_phi(1, R, H, beta, 1, N, N)
    mconp = mglb.dmoments(L, sconp)
    assert (np.abs(conp - mconp) < 0.2)[:3].all()
