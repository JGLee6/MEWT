# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:35:15 2020

@author: jgl6
"""
import numpy as np
import mewt.maglib as mglb
import mewt.maglibShapes as mshp
import newt.multipoleLib as mplb
import newt.translations as trs
import newt.rotations as rot


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


def test_rot_a():
    """
    GIVEN unit-dipole translated unit distance along axis, dipole along axis
    WHEN rotated to other axis
    THEN moment calculations match to l=10
    """
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
    # y-dip at origin -> mom -> [0, 1, 0]
    magy = np.array([[1, 0, 0, 0, 1, 0, 1, 0]])
    dmomy = mglb.dmoments(lmax, magy)
    dmomyt = trs.translate_qlm(dmomy, [0, 1, 0])
    # y-dip at [0, 1, 0] -> mom
    magty = np.array([[1, 0, 1, 0, 1, 0, 1, 0]])
    dmomty = mglb.dmoments(lmax, magty)
    assert (np.abs(dmomty - dmomyt) < 2e2*np.finfo(float).eps).all()
    # Check if rotations to others match
    # z-dip at [0, 0, 1] -> rotate z to x-axis
    dmomxrt = rot.rotate_qlm(dmomzt, 0, np.pi/2, 0)
    assert (np.abs(dmomxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
    # z-dip at [0, 0, 1] -> rotate z to y-axis
    dmomyrt = rot.rotate_qlm(dmomzt, np.pi/2, np.pi/2, 0)
    assert (np.abs(dmomyrt - dmomyt) < 2e2*np.finfo(float).eps).all()
    # x-dip at [1, 0, 0] -> rotate x to y-axis
    dmomxyrt = rot.rotate_qlm(dmomxt, np.pi/2, 0, 0)
    assert (np.abs(dmomxyrt - dmomyt) < 2e2*np.finfo(float).eps).all()
    # x-dip at [1, 0, 0] -> rotate x to z-axis
    dmomxzrt = rot.rotate_qlm(dmomxt, 0, -np.pi/2, 0)
    assert (np.abs(dmomxzrt - dmomzt) < 2e2*np.finfo(float).eps).all()
    # y-dip at [0, 1, 0] -> rotate y to x-axis
    dmomyxrt = rot.rotate_qlm(dmomyt, -np.pi/2, 0, 0)
    assert (np.abs(dmomyxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
    # y-dip at [0, 1, 0] -> rotate y to z-axis
    dmomyzrt = rot.rotate_qlm(dmomyt, np.pi/2, np.pi/2, np.pi/2)
    assert (np.abs(dmomyzrt - dmomzt) < 2e2*np.finfo(float).eps).all()


def test_rot_b():
    """
    GIVEN unit-dipole translated unit distance along axis, dipole perp to axis
    WHEN rotated to other axis
    THEN moment calculations match to l=10

    Moments are named identical to test_rot_a with values swapped and
    appropriate rotations substituted.
    """
    lmax = 10
    # x-dip at origin -> mom -> [0, 0, 1]
    magz = np.array([[1, 0, 0, 0, 1, 1, 0, 0]])
    dmomz = mglb.dmoments(lmax, magz)
    dmomzt = trs.translate_qlm(dmomz, [0, 0, 1])
    # x-dip at [0, 0, 1] -> mom
    magtz = np.array([[1, 0, 0, 1, 1, 1, 0, 0]])
    dmomtz = mglb.dmoments(lmax, magtz)
    assert (np.abs(dmomtz - dmomzt) < 2e2*np.finfo(float).eps).all()
    # y-dip at origin -> mom -> [1, 0, 0]
    magx = np.array([[1, 0, 0, 0, 1, 0, 1, 0]])
    dmomx = mglb.dmoments(lmax, magx)
    dmomxt = trs.translate_qlm(dmomx, [1, 0, 0])
    # y-dip at [1, 0, 0] -> mom
    magtx = np.array([[1, 1, 0, 0, 1, 0, 1, 0]])
    dmomtx = mglb.dmoments(lmax, magtx)
    assert (np.abs(dmomtx - dmomxt) < 2e2*np.finfo(float).eps).all()
    # z-dip at origin -> mom -> [0, 1, 0]
    magy = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    dmomy = mglb.dmoments(lmax, magy)
    dmomyt = trs.translate_qlm(dmomy, [0, 1, 0])
    # z-dip at [0, 1, 0] -> mom
    magty = np.array([[1, 0, 1, 0, 1, 0, 0, 1]])
    dmomty = mglb.dmoments(lmax, magty)
    assert (np.abs(dmomty - dmomyt) < 2e2*np.finfo(float).eps).all()
    # Check if rotations to others match
    # x-dip at [0, 0, 1] -> rotate z to x-axis
    dmomxrt = rot.rotate_qlm(dmomzt, 0, np.pi/2, np.pi/2)
    assert (np.abs(dmomxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
    # x-dip at [0, 0, 1] -> rotate z to y-axis
    dmomyrt = rot.rotate_qlm(dmomzt, np.pi/2, np.pi/2, np.pi)
    assert (np.abs(dmomyrt - dmomyt) < 2e2*np.finfo(float).eps).all()
    # y-dip at [1, 0, 0] -> rotate x to y-axis
    dmomxyrt = rot.rotate_qlm(dmomxt, 0, np.pi/2, np.pi/2)
    assert (np.abs(dmomxyrt - dmomyt) < 2e2*np.finfo(float).eps).all()
    # y-dip at [1, 0, 0] -> rotate x to z-axis
    dmomxzrt = rot.rotate_qlm(dmomxt, -np.pi/2, -np.pi/2, 0)
    # slightly higher error here
    assert (np.abs(dmomxzrt - dmomzt) < 3e2*np.finfo(float).eps).all()
    # z-dip at [0, 1, 0] -> rotate y to x-axis
    dmomyxrt = rot.rotate_qlm(dmomyt, -np.pi/2, -np.pi/2, 0)
    assert (np.abs(dmomyxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
    # z-dip at [0, 1, 0] -> rotate y to z-axis
    dmomyzrt = rot.rotate_qlm(dmomyt, 0, np.pi/2, np.pi/2)
    assert (np.abs(dmomyzrt - dmomzt) < 2e2*np.finfo(float).eps).all()


def test_rot_0():
    """
    GIVEN a z-dipole at the origin
    WHEN rotated 100x at random 3d angle
    THEN ensure dipole magnitude unchaged.
    """
    # z-dip at origin -> mom -> [0, 0, 1]
    magz = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    for k in range(100):
        rands = np.random.rand(4)
        rands[0] *= np.pi*2
        magz = mglb.rotate_dipole_array(magz, rands[0], rands[1:])
    # Check that magnitude of magnetization unchanged
    assert magz[0, 4] == 1
    # Check that magnitude of magnetization vector unchanged
    svec = magz[0, 5:]
    assert np.abs(np.dot(svec, svec) - 1) < 1e3*np.finfo(float).eps


def test_ft_a():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis both with
    vertical unit dipole moments.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    mag2 = np.array([[1, 1, 0, 0, 1, 0, 0, 1]])
    fpred = np.array([-3*mglb.mu_0/(4*np.pi), 0, 0])
    tpred = np.array([0, 0, 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    # minus sign on force?
    force = -force*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    assert (np.abs(tpred[2] - torque) < 10*np.finfo(float).eps).all()


def test_ft_b():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis, one at
    the origin with vertical unit dipole moment and the other with x-oriented
    unit dipole moment.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1]])
    mag2 = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    fpred = np.array([0, 0, -3*mglb.mu_0/(4*np.pi)])
    tpred = np.array([0, 2*mglb.mu_0/(4*np.pi), 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t2 - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    # minus sign on force?
    force = -force*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    assert (np.abs(tpred[2] - torque) < 10*np.finfo(float).eps).all()


def test_ft_c():
    """
    Compares the analytic force and torque prediction to the point_dipole
    calculation of two dipoles separated by a meter along the x-axis, both with
    x-oriented unit dipole moments.
    """
    mag1 = np.array([[1, 0, 0, 0, 1, 1, 0, 0]])
    mag2 = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    fpred = np.array([6*mglb.mu_0/(4*np.pi), 0, 0])
    tpred = np.array([0, 0, 0])
    f, t2, t = mglb.point_matrix_magnets(mag1, mag2)
    assert (np.abs(f - fpred) < 10*np.finfo(float).eps).all()
    assert (np.abs(t - tpred) < 10*np.finfo(float).eps).all()
    lmax = 10
    dmom = mglb.dmoments(lmax, mag1)
    Dmom = mglb.Dmomentsb(lmax, mag2)
    force = mplb.multipole_force(lmax, dmom, Dmom, 0, 0, 0)
    # minus sign on force?
    force = -force*mglb.magC/mplb.BIG_G
    tqlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmom)
    torque = np.real(np.sum(tqlm))*mglb.magC/mplb.BIG_G
    assert (np.abs(fpred - force) < 10*np.finfo(float).eps).all()
    assert (np.abs(tpred[2] - torque) < 10*np.finfo(float).eps).all()


def test_Dmombz():
    """
    Compares outer moment of z-dipole at the [0, 0, 10] + translate [0, 0, .5]
    to outer moment of z-dipole at [0, 0, 10.5]. And similar for translations
    in x, y.
    """
    mag1 = np.array([[1, 0, 0, 10, 1, 0, 0, 1]])
    lmax = 10
    # Test z translation
    dz = [0, 0, .5]
    # outer moments -> translate dz
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dz)
    # translate dz -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dz)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test x translation
    dx = [.5, 0, 0]
    # outer moments -> translate dx
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dx)
    # translate dx -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dx)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test y translation
    dy = [0, .5, 0]
    # outer moments -> translate dy
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dy)
    # translate dy -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dy)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()


def test_Dmombx():
    """
    Compares outer moment of x-dipole at the [10, 0, 0] + translate [.5, 0, 0]
    to outer moment of x-dipole at [10.5, 0, 0]. And similar for translations
    in z, y.
    """
    mag1 = np.array([[1, 10, 0, 0, 1, 1, 0, 0]])
    lmax = 10
    # Test z translation
    dz = [0, 0, .5]
    # outer moments -> translate dz
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dz)
    # translate dz -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dz)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test x translation
    dx = [.5, 0, 0]
    # outer moments -> translate dx
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dx)
    # translate dx -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dx)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test y translation
    dy = [0, .5, 0]
    # outer moments -> translate dy
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dy)
    # translate dy -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dy)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()


def test_Dmomby():
    """
    Compares outer moment of y-dipole at the [0, 10, 0] + translate [0, .5, 0]
    to outer moment of y-dipole at [0, 10.5, 0]. And similar for translations
    in z, x.
    """
    mag1 = np.array([[1, 0, 10, 0, 1, 0, 1, 0]])
    lmax = 10
    # Test z translation
    dz = [0, 0, .5]
    # outer moments -> translate dz
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dz)
    # translate dz -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dz)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test x translation
    dx = [.5, 0, 0]
    # outer moments -> translate dx
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dx)
    # translate dx -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dx)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()
    # Test y translation
    dy = [0, .5, 0]
    # outer moments -> translate dy
    DmomA = mglb.Dmomentsb(lmax, mag1)
    DmomA = trs.translate_Qlmb(DmomA, dy)
    # translate dy -> outer moments
    mag1z = mglb.translate_dipole_array(mag1, dy)
    DmomB = mglb.Dmomentsb(lmax, mag1z)
    assert (np.abs(DmomA - DmomB) < 1e6*np.finfo(float).eps).all()


def test_Dmombs_dz():
    """
    Creates a pair of point dipole at [1, 0, 0] and [-1, 0, 0]. Then compares
    the outer moments of the vertically translated ([0, 0, 5]) points computed
    in two ways. The first method is through translating the points and
    computing the outer moments of point dipoles. The second method, computes
    the inner moments of point dipoles and translates to outer moments using a
    translation method of inner->outer moments.
    """
    mag2 = np.array([[1, 1, 0, 0, 1, 0, 0, 1], [1, -1, 0, 0, 1, 0, 0, 1]])
    lmax = 20
    dz = [0, 0, 5]
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dz)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dz)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [1, -1, 0, 0, 1, -1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dz)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dz)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 1, 0, 0, 1, 0, 1, 0], [1, -1, 0, 0, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dz)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dz)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()


def test_Dmombs_dx():
    """
    Creates a pair of point dipole at [0, 0, 1] and [0, 0, -1]. Then compares
    the outer moments of the vertically translated ([5, 0, 0]) points computed
    in two ways. The first method is through translating the points and
    computing the outer moments of point dipoles. The second method, computes
    the inner moments of point dipoles and translates to outer moments using a
    translation method of inner->outer moments.
    """
    mag2 = np.array([[1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, -1, 1, 0, 0, 1]])
    lmax = 20
    dx = [5, 0, 0]
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 0, -1, 1, -1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, -1, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()


def test_Dmombs_dy():
    """
    Creates a pair of point dipole at [0, 0, 1] and [0, 0, -1]. Then compares
    the outer moments of the vertically translated ([0, 5, 0]) points computed
    in two ways. The first method is through translating the points and
    computing the outer moments of point dipoles. The second method, computes
    the inner moments of point dipoles and translates to outer moments using a
    translation method of inner->outer moments.
    """
    mag2 = np.array([[1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, -1, 1, 0, 0, 1]])
    lmax = 20
    dx = [0, 5, 0]
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 0, -1, 1, -1, 0, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()
    mag2 = np.array([[1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, -1, 1, 0, 1, 0]])
    dmom = mglb.dmoments(lmax, mag2)
    mag3 = mglb.translate_dipole_array(mag2, dx)
    DmomA = mglb.Dmomentsb(lmax, mag3)
    DmomB = trs.translate_q2Q(dmom, dx)
    assert (np.abs(DmomA - DmomB) < 5e6*np.finfo(float).eps).all()


def test_force_x():
    """
    GIVEN 3-fold x-dipoles in circle of radius=1, at phi=0 and z=0
    AND 3-fold x-dipoles in circle of radius=1 at phi=pi/6 and z=3
    WHEN force and torques are computed using maglib.point_matrix_magnets
    AND using an explicit write of summing over all 9 pairs
    AND by converting to multipole moments and computing force and torques
    THEN expect all three match to roughly floating point precision
    """
    thetak = np.pi/6
    z = 3
    lmax = 20
    mag3p = xpoint(0, 3)
    mag3a = xpoint(thetak, 3)
    mag3a = mglb.translate_dipole_array(mag3a, [0, 0, z])
    frc, trq2, trq = mglb.point_matrix_magnets(mag3p, mag3a)
    frcb, trq2b, trqb = 0, 0, 0
    for k in range(len(mag3p)):
        for m in range(len(mag3a)):
            frckm, trqkm = mglb.mag_ft_array(mag3p[k], mag3a[m])
            frcb += frckm
            trq2b += trqkm
            trqb += np.cross(mag3p[k, 1:4], frckm)
    dmom = mglb.dmoments(lmax, mag3p)
    Dmomb = mglb.Dmomentsb(lmax, mag3a)
    frcc = mplb.multipole_force(lmax, dmom, Dmomb, 0, 0, 0)
    frcc *= -mglb.magC/mplb.BIG_G
    tlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmomb)
    tc *= -mglb.magC/mplb.BIG_G
    assert (np.abs(frc - frcb) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq2 - trq2b) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq - trqb) < 10*np.finfo(float).eps).all()
    assert (np.abs(frc - frcc) < 1e2*np.finfo(float).eps).all()
    assert (np.abs(trq2[2]+trq[2] + np.sum(tc)) < 10*np.finfo(float).eps).all()


def test_force_y():
    """
    GIVEN 3-fold y-dipoles in circle of radius=1, at phi=0 and z=0
    AND 3-fold y-dipoles in circle of radius=1 at phi=pi/6 and z=3
    WHEN force and torques are computed using maglib.point_matrix_magnets
    AND using an explicit write of summing over all 9 pairs
    AND by converting to multipole moments and computing force and torques
    THEN expect all three match to roughly floating point precision
    """
    thetak = np.pi/6
    z = 3
    lmax = 20
    mag3p = ypoint(0, 3)
    mag3a = ypoint(thetak, 3)
    mag3a = mglb.translate_dipole_array(mag3a, [0, 0, z])
    frc, trq2, trq = mglb.point_matrix_magnets(mag3p, mag3a)
    frcb, trq2b, trqb = 0, 0, 0
    for k in range(len(mag3p)):
        for m in range(len(mag3a)):
            frckm, trqkm = mglb.mag_ft_array(mag3p[k], mag3a[m])
            frcb += frckm
            trq2b += trqkm
            trqb += np.cross(mag3p[k, 1:4], frckm)
    dmom = mglb.dmoments(lmax, mag3p)
    Dmomb = mglb.Dmomentsb(lmax, mag3a)
    frcc = mplb.multipole_force(lmax, dmom, Dmomb, 0, 0, 0)
    frcc *= -mglb.magC/mplb.BIG_G
    tlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmomb)
    tc *= -mglb.magC/mplb.BIG_G
    assert (np.abs(frc - frcb) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq2 - trq2b) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq - trqb) < 10*np.finfo(float).eps).all()
    assert (np.abs(frc - frcc) < 1e2*np.finfo(float).eps).all()
    assert (np.abs(trq2[2]+trq[2] + np.sum(tc)) < 10*np.finfo(float).eps).all()


def test_force_z():
    """
    GIVEN 3-fold z-dipoles in circle of radius=1, at phi=0 and z=0
    AND 3-fold z-dipoles in circle of radius=1 at phi=pi/6 and z=3
    WHEN force and torques are computed using maglib.point_matrix_magnets
    AND using an explicit write of summing over all 9 pairs
    AND by converting to multipole moments and computing force and torques
    THEN expect all three match to roughly floating point precision
    """
    thetak = np.pi/6
    z = 3
    lmax = 20
    mag3p = zpoint(0, 3)
    mag3a = zpoint(thetak, 3)
    mag3a = mglb.translate_dipole_array(mag3a, [0, 0, z])
    frc, trq2, trq = mglb.point_matrix_magnets(mag3p, mag3a)
    frcb, trq2b, trqb = 0, 0, 0
    for k in range(len(mag3p)):
        for m in range(len(mag3a)):
            frckm, trqkm = mglb.mag_ft_array(mag3p[k], mag3a[m])
            frcb += frckm
            trq2b += trqkm
            trqb += np.cross(mag3p[k, 1:4], frckm)
    dmom = mglb.dmoments(lmax, mag3p)
    Dmomb = mglb.Dmomentsb(lmax, mag3a)
    frcc = mplb.multipole_force(lmax, dmom, Dmomb, 0, 0, 0)
    frcc *= -mglb.magC/mplb.BIG_G
    tlm, tc, ts = mplb.torque_lm(lmax, dmom, Dmomb)
    tc *= -mglb.magC/mplb.BIG_G
    assert (np.abs(frc - frcb) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq2 - trq2b) < 10*np.finfo(float).eps).all()
    assert (np.abs(trq - trqb) < 10*np.finfo(float).eps).all()
    assert (np.abs(frc - frcc) < 1e2*np.finfo(float).eps).all()
    assert (np.abs(trq2[2]+trq[2] + np.sum(tc)) < 10*np.finfo(float).eps).all()


def energy_3om(theta, z):
    """
    Assumes triplets of only x-oriented dipoles at radius of 1m with separation
    z. The three-fold attractor dipoles are rotated by an angle theta. Returns
    the energy
    """
    U = 0
    z = z*np.ones(len(theta))
    for k in range(3):
        for m in range(3):
            x = np.cos(2*np.pi*k/3)-np.cos(2*np.pi*m/3+theta)
            y = np.sin(2*np.pi*k/3)-np.sin(2*np.pi*m/3+theta)
            rvec = np.stack([x, y, z])
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
