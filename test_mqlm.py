#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:36:17 2022

@author: John Greendeer Lee
"""
import numpy as np
import mqlm
import maglib as mglb
import maglibShapes as mshp
import newt.rotations as rot


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
    # Compare two methods
    rqlmx = mqlm.rect_prism_x(10, 1, 3, 2, 5, 0)
    rqlmx2 = mqlm.rect_prism_x2(10, 1, 3, 2, 5, 0)
    assert (np.abs(rqlmx2 - rqlmx) < 3e5*np.finfo(float).eps).all()
    # Compare to z-oriented rotated to x-oriented
    rqlm0 = mqlm.rect_prism_z(10, 1, 2, 3, 5, 0)
    rqlmx3 = rot.rotate_qlm(rqlm0, 0, np.pi/2, 0)
    assert (np.abs(rqlmx2 - rqlmx3) < 3e5*np.finfo(float).eps).all()
    # Compare to rect assembled from triangular prisms
    tri0 = mqlm.tri_prism_x(10, 1, 3, 2/2, -2.5, 2.5)
    tri1 = mqlm.tri_prism_x(10, -1, 3, 2/2, -2.5, 2.5)
    tri1 = rot.rotate_qlm(tri1, 0, 0, np.pi)
    tri2 = mqlm.tri_prism_y(10, 1, 3, 5/2, -1, 1)
    tri3 = mqlm.tri_prism_y(10, -1, 3, 5/2, -1, 1)
    tri2 = rot.rotate_qlm(tri2, 0, 0, -np.pi/2)
    tri3 = rot.rotate_qlm(tri3, 0, 0, np.pi/2)
    rqlmx4 = tri0 + tri1 + tri2 + tri3
    assert (np.abs(rqlmx2 - rqlmx4) < 3e5*np.finfo(float).eps).all()
    # Compare to point-dipole approximation to l=5
    N = 40
    rect5 = mshp.rectangle(1, 2, 5, 3, 1, 1, 0, 0, N, N, N)
    rqlmx5 = mglb.dmoments(10, rect5)
    assert (np.abs(rqlmx2 - rqlmx5) < .1)[:5].all()


def test_recy():
    # Compare two methods
    rqlmy = mqlm.rect_prism_y(10, 1, 5, 3, 2, 0)
    rqlmy2 = mqlm.rect_prism_y2(10, 1, 5, 3, 2, 0)
    assert(np.abs(rqlmy-rqlmy2) < 3e5*np.finfo(float).eps).all()
    # Compare to z-oriented rotated to y-oriented
    rqlm0 = mqlm.rect_prism_z(10, 1, 2, 3, 5, 0)
    rqlmy3 = rot.rotate_qlm(rqlm0, np.pi/2, np.pi/2, -np.pi/2)
    assert (np.abs(rqlmy2 - rqlmy3) < 3e5*np.finfo(float).eps).all()
    # Compare to rect assembled from triangular prisms
    tri0 = mqlm.tri_prism_y(10, 1, 5, 3/2, -1, 1)
    tri1 = mqlm.tri_prism_y(10, -1, 5, 3/2, -1, 1)
    tri1 = rot.rotate_qlm(tri1, 0, 0, np.pi)
    tri2 = mqlm.tri_prism_x(10, 1, 5, 2/2, -1.5, 1.5)
    tri3 = mqlm.tri_prism_x(10, -1, 5, 2/2, -1.5, 1.5)
    tri2 = rot.rotate_qlm(tri2, 0, 0, np.pi/2)
    tri3 = rot.rotate_qlm(tri3, 0, 0, -np.pi/2)
    rqlmy4 = tri0 + tri1 + tri2 + tri3
    assert (np.abs(rqlmy4 - rqlmy3) < 3e5*np.finfo(float).eps).all()
    # Compare to point-dipole approximation to l=5
    N = 40
    rect5 = mshp.rectangle(1, 3, 2, 5, 1, 0, 1, 0, N, N, N)
    rqlmy5 = mglb.dmoments(10, rect5)
    assert (np.abs(rqlmy5 - rqlmy2) < .1)[:5].all()


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
    annx = mqlm.annulus_x(L, 1, H, IR, OR, 0, beta)
    anny = mqlm.annulus_y(L, 1, H, IR, OR, 0, beta)
    annx2 = rot.rotate_qlm(anny, 0, 0, -np.pi/2)
    assert (np.abs(annx - annx2) < 3e4*np.finfo(float).eps).all()
    sannx = mshp.wedge(1, IR, OR, H, beta, 1, 1, 0, 0, N, N)
    mannx = mglb.dmoments(L, sannx)
    assert (np.abs(annx - mannx) < 0.1)[:3].all()
    sanny = mshp.wedge(1, IR, OR, H, beta, 1, 0, 1, 0, N, N)
    manny = mglb.dmoments(L, sanny)
    assert (np.abs(anny - manny) < 0.1)[:3].all()
    annz = mqlm.annulus_z(L, 1, H, IR, OR, 0, beta)
    sannz = mshp.annulus(1, IR, OR, H, 1, 0, 0, 1, N, N)
    mannz = mglb.dmoments(L, sannz)
    assert (np.abs(annz - mannz) < 0.2)[:3].all()
    annr = mqlm.annulus_r(L, 1, H, IR, OR, 0, beta)
    sannr = mshp.wedge_rho(1, IR, OR, H, beta, 1, N, N)
    mannr = mglb.dmoments(L, sannr)
    assert (np.abs(annr - mannr) < 0.1)[:3].all()
    annp = mqlm.annulus_p(L, 1, H, IR, OR, 0, beta)
    sannp = mshp.wedge_phi(1, IR, OR, H, beta, 1, N, N)
    mannp = mglb.dmoments(L, sannp)
    assert (np.abs(annp - mannp) < 0.1)[:3].all()


def test_ann2():
    H = 3
    IR = 1.5
    OR = 2
    L = 10
    N = 40
    beta = np.pi/6
    annx = mqlm.annulus_x(L, 1, H, IR, OR, 0, beta)
    anny = mqlm.annulus_y(L, 1, H, IR, OR, 0, beta)
    # annx2 = rot.rotate_qlm(anny, 0, 0, -np.pi/2)
    # assert (np.abs(annx - annx2) < 3e4*np.finfo(float).eps).all()
    sannx = mshp.wedge(1, IR, OR, H, beta, 1, 1, 0, 0, N, N)
    mannx = mglb.dmoments(L, sannx)
    assert (np.abs(annx - mannx) < 0.1)[:3].all()
    sanny = mshp.wedge(1, IR, OR, H, beta, 1, 0, 1, 0, N, N)
    manny = mglb.dmoments(L, sanny)
    assert (np.abs(anny - manny) < 0.1)[:3].all()
    annz = mqlm.annulus_z(L, 1, H, IR, OR, 0, beta)
    sannz = mshp.wedge(1, IR, OR, H, beta, 1, 0, 0, 1, N, N)
    mannz = mglb.dmoments(L, sannz)
    assert (np.abs(annz - mannz) < 0.2)[:3].all()
    annr = mqlm.annulus_r(L, 1, H, IR, OR, 0, beta)
    sannr = mshp.wedge_rho(1, IR, OR, H, beta, 1, N, N)
    mannr = mglb.dmoments(L, sannr)
    assert (np.abs(annr - mannr) < 0.1)[:3].all()
    annp = mqlm.annulus_p(L, 1, H, IR, OR, 0, beta)
    sannp = mshp.wedge_phi(1, IR, OR, H, beta, 1, N, N)
    mannp = mglb.dmoments(L, sannp)
    assert (np.abs(annp - mannp) < 0.1)[:3].all()


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
    scony = mshp.cone(1, R, H, beta, 1, 0, 1, 0, N, N)
    mcony = mglb.dmoments(L, scony)
    assert (np.abs(cony - mcony) < 0.1)[:3].all()
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


def test_cone2():
    H = 3
    R = 2
    L = 10
    N = 30
    beta = np.pi/6
    conx = mqlm.cone_x(L, 1, H, R, 0, beta)
    cony = mqlm.cone_y(L, 1, H, R, 0, beta)
    # conx2 = rot.rotate_qlm(cony, 0, 0, -np.pi/2)
    # assert (np.abs(conx - conx2) < 3e3*np.finfo(float).eps).all()
    sconx = mshp.cone(1, R, H, beta, 1, 1, 0, 0, N, N)
    mconx = mglb.dmoments(L, sconx)
    assert (np.abs(conx - mconx) < 0.1)[:3].all()
    scony = mshp.cone(1, R, H, beta, 1, 0, 1, 0, N, N)
    mcony = mglb.dmoments(L, scony)
    assert (np.abs(cony - mcony) < 0.1)[:3].all()
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
