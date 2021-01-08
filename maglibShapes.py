# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:51:39 2020

@author: John Greendeer Lee
"""
import numpy as np


def rectangle(mass, x, y, z, s, sx, sy, sz, nx, ny, nz):
    """
    Creates point masses distributed in an rectangular solid of mass m.

    Inputs
    ------
    mass : float
        mass in kg
    x : float
        x-length of brick in m
    y : float
        y-length of brick in m
    z : float
        z-length of brick in m
    s : float
        Magnetism density (A/m)
    sx : float
        xhat portion of magnetism density direction, |sx| <= 1
    sy : float
        yhat portion of magnetism density direction, |sy| <= 1
    sz : float
        zhat portion of magnetism density direction, |sz| <= 1
    nx : float
        number of points distributed in x
    nz : float
        number of points distributed in y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        nx*ny*nz x 8 point mass array of format [m, x, y, z, s, sx, sy, sz]
    """
    pointArray = np.zeros([nx*ny*nz, 8])
    zgrid = z/float(nz)
    xgrid = x/float(nx)
    ygrid = y/float(ny)
    pointArray[:, 0] = mass/float(nx*ny*nz)
    pointArray[:, 4] = s*x*y*z/float(nx*ny*nz)
    pointArray[:, 5:] = sx, sy, sz

    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*ny*nx+l*nx+m, 1] = (m-(nx-1)/2)*xgrid
                pointArray[k*ny*nx+l*nx+m, 2] = (l-(ny-1)/2)*ygrid
                pointArray[k*ny*nx+l*nx+m, 3] = (k-(nz-1)/2)*zgrid

    return pointArray


def annulus(mass, iR, oR, t, s, sx, sy, sz, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = t/float(nz)
    xgrid = oR*2./float(nx)
    ygrid = xgrid

    boxvol = 4*t*oR**2
    vol = np.pi*(oR**2 - iR**2)*t
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    loopCounter = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                pointArray[loopCounter, 1] = (m-(nx-1)/2)*xgrid
                pointArray[loopCounter, 2] = (l-(nx-1)/2)*ygrid
                pointArray[loopCounter, 3] = (k-(nz-1)/2)*zgrid
                loopCounter += 1

    pointArray = np.array([pointArray[k] for k in range(nx*nx*nz) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 >= iR**2 and
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <= oR**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nz*nx*nx)
    pointArray[:, 5:] = sx, sy, sz

    return pointArray


def wedge(mass, iR, oR, t, beta, s, sx, sy, sz, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = oR
    if beta < np.pi/2:
        xmin = np.cos(beta)*iR
        ymax = np.sin(beta)*oR
    else:
        xmin = np.cos(beta)*oR
        ymax = oR
    xave = (xmax+xmin)/2
    zgrid = t/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = t*(xmax-xmin)*ymax*2
    vol = beta*(oR**2-iR**2)*t
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= oR and r >= iR:
                    pointArray[ctr, 1:4] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nz*nx*nx)
    pointArray[:, 5:] = sx, sy, sz

    return pointArray


def wedge_rho(mass, iR, oR, t, beta, s, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m with radially-
    polarized spin density, s.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = oR
    if beta < np.pi/2:
        xmin = np.cos(beta)*iR
        ymax = np.sin(beta)*oR
    else:
        xmin = np.cos(beta)*oR
        ymax = oR
    xave = (xmax+xmin)/2
    zgrid = t/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = t*(xmax-xmin)*ymax*2
    vol = beta*(oR**2-iR**2)*t
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= oR and r >= iR:
                    rhohatx, rhohaty = x/r, y/r
                    pointArray[ctr, 1:4] = [x, y, z]
                    pointArray[ctr, 4:] = [s, rhohatx, rhohaty, 0]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    # pointArray[:, 4] = s*boxvol/(nz*nx*nx)
    # pointArray[:, 5:] = sx, sy, sz

    return pointArray


def wedge_phi(mass, iR, oR, t, beta, s, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m with azimuthally-
    polarized spin density, s.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = oR
    if beta < np.pi/2:
        xmin = np.cos(beta)*iR
        ymax = np.sin(beta)*oR
    else:
        xmin = np.cos(beta)*oR
        ymax = oR
    xave = (xmax+xmin)/2
    zgrid = t/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = t*(xmax-xmin)*ymax*2
    vol = beta*(oR**2-iR**2)*t
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= oR and r >= iR:
                    phihatx, phihaty = -y/r, x/r
                    pointArray[ctr, 1:4] = [x, y, z]
                    pointArray[ctr, 4:] = [s, phihatx, phihaty, 0]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    # pointArray[:, 4] = s*boxvol/(nz*nx*nx)
    # pointArray[:, 5:] = sx, sy, sz

    return pointArray


def cone(mass, R, H, beta, s, sx, sy, sz, nx, nz):
    """
    Creates point masses distributed in a section of a cone of mass m and spin-
    density s.

    Inputs
    ------
    m : float
        mass in kg
    R : float
        radius of cone section in m
    H : float
        height of cone above xy-plane in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = R
    if beta < np.pi/2:
        xmin = 0
        ymax = np.sin(beta)*R
    else:
        xmin = np.cos(beta)*R
        ymax = R
    xave = (xmax+xmin)/2
    zgrid = H/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = H*(xmax-xmin)*ymax*2
    vol = H*beta*R**2/3
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid + H/2
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= R and r**2 <= R**2*(1-z/H)**2:
                    pointArray[ctr, 1:4] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nz*nx*nx)
    pointArray[:, 5:] = sx, sy, sz

    return pointArray


def cone_rho(mass, R, H, beta, s, nx, nz):
    """
    Creates point masses distributed in a section of a cone of mass m and spin-
    density s oriented radially from z-axis.

    Inputs
    ------
    m : float
        mass in kg
    R : float
        radius of cone section in m
    H : float
        height of cone above xy-plane in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = R
    if beta < np.pi/2:
        xmin = 0
        ymax = np.sin(beta)*R
    else:
        xmin = np.cos(beta)*R
        ymax = R
    xave = (xmax+xmin)/2
    zgrid = H/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = H*(xmax-xmin)*ymax*2
    vol = H*beta*R**2/3
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid + H/2
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= R and r**2 <= R**2*(1-z/H)**2:
                    if r != 0:
                        sr, rhohatx, rhohaty = s, x/r, y/r
                    else:
                        sr, rhohatx, rhohaty = 0, 0, 0
                    pointArray[ctr, 1:4] = [x, y, z]
                    pointArray[ctr, 4:] = [sr, rhohatx, rhohaty, 0]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nz*nx*nx)

    return pointArray


def cone_phi(mass, R, H, beta, s, nx, nz):
    """
    Creates point masses distributed in a section of a cone of mass m and spin-
    density s oriented azimuthally around z-axis.

    Inputs
    ------
    m : float
        mass in kg
    R : float
        radius of cone section in m
    H : float
        height of cone above xy-plane in m
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = R
    if beta < np.pi/2:
        xmin = 0
        ymax = np.sin(beta)*R
    else:
        xmin = np.cos(beta)*R
        ymax = R
    xave = (xmax+xmin)/2
    zgrid = H/nz
    xgrid = (xmax-xmin)/nx
    ygrid = 2*ymax/nx

    boxvol = H*(xmax-xmin)*ymax*2
    vol = H*beta*R**2/3
    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 8])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid + H/2
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= R and r**2 <= R**2*(1-z/H)**2:
                    if r != 0:
                        sp, phihatx, phihaty = s, -y/r, x/r
                    else:
                        sp, phihatx, phihaty = 0, 0, 0
                    pointArray[ctr, 1:4] = [x, y, z]
                    pointArray[ctr, 4:] = [sp, phihatx, phihaty, 0]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nz*nx*nx)

    return pointArray


def tri_prism(mass, d, y1, y2, t, s, sx, sy, sz, nx, ny, nz):
    """
    Creates point masses distributed in a triangular prism of mass m.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    s : float
        Magnetism density (A/m)
    sx : float
        xhat portion of magnetism density direction, |sx| <= 1
    sy : float
        yhat portion of magnetism density direction, |sy| <= 1
    sz : float
        zhat portion of magnetism density direction, |sz| <= 1
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    if y2 < y1:
        print('Require y2 > y1')
        return []
    base = np.max([0, y2])-np.min([0, y1])
    yave = (y2+y1)/2
    zgrid = t/nz
    xgrid = d/nx
    ygrid = base/ny

    boxvol = t*base*d
    vol = t*(y2-y1)*d/2
    density = mass/vol
    pointMass = density*(xgrid*ygrid*zgrid/2)

    pointArray = np.zeros([nx*ny*nz, 8])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*nx*ny+l*nx+m, 1] = (m-(nx-1)/2)*xgrid + d/2
                pointArray[k*nx*ny+l*nx+m, 2] = (l-(ny-1)/2)*ygrid + yave
                pointArray[k*nx*ny+l*nx+m, 3] = (k-(nz-1)/2)*zgrid

    pointArray = np.array([pointArray[k] for k in range(nx*ny*nz) if
                           pointArray[k, 1]*y1 <= pointArray[k, 2]*d and
                           pointArray[k, 1]*y2 >= pointArray[k, 2]*d])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    pointArray[:, 4] = s*boxvol/(nx*ny*nz)
    pointArray[:, 5:] = sx, sy, sz

    return pointArray
