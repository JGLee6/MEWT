# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:57:55 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def rect_prism_z(L, Mz, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the z-axis with a magnetization density Mz.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mz : float
        Vertical magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mz*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m odd, m even -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m even
        for m in range(0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m-2*k+1)
                slk = (-1)**k*H**(l-2*k-m)/2**(l+2*k+m-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        # j+p even
                        if (jp % 2) == 0:
                            kfac = a**(2*k+m-jp+1)*b**(jp+1)
                            ksum += 1j**jp*sp.comb(k, j)*kfac/((m+2*k-jp+1)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism_z2(L, Mz, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the z-axis with a magnetization density Mz.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mz : float
        Vertical magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mz*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m odd, m even -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m even
        for m in range(0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m-2*k+1)
                slk = (-1)**k*H**(l-2*k-m)/2**(l+2*k+m-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        # j+p even
                        if (jp % 2) == 0:
                            kfac = a**(2*k+m-jp+1)*b**(jp+1)
                            kfac += (-1)**(m//2)*b**(2*k+m-jp+1)*a**(jp+1)
                            ksum += 1j**(jp)*sp.comb(k, j)*kfac/((m+2*k+2)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism_x(L, Mx, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the x-axis with a magnetization density Mx.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mx : float
        Horizontal magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mx*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m even, m odd -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m odd
        for m in range(1, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = m+2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**(m+k)*H**(l-m2k+1)/2**(l+m2k-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        # j+p even
                        if (jp % 2) == 0:
                            kfac = a**(m2k-jp)*b**(jp+1)
                            ksum += 1j**jp*sp.comb(k, j)*kfac/(jp+1)
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism_x2(L, Mx, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the x-axis with a magnetization density Mx.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mx : float
        Horizontal magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = -Mx*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m even, m odd -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m odd
        for m in range(1, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = m+2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**k*H**(l-m2k+1)/2**(l+m2k-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        if (jp % 2) == 0:
                            kfac = (m2k-jp)*a**(m2k-jp)*b**(jp+1)
                        else:
                            kfac = 1j**m*(jp+1)*b**(m2k-jp+1)*a**(jp)
                        ksum += 1j**(jp)*sp.comb(k, j)*kfac/((m2k+1)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism_y(L, My, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the y-axis with a magnetization density Mx.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    My : float
        Horizontal magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = My*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m even, m odd -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m odd
        for m in range(1, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = m+2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**(m+k)*H**(l-m2k+1)/2**(l+m2k-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        # j+p odd
                        if (jp % 2) == 1:
                            kfac = a**(m2k-jp+1)*b**(jp)
                            ksum += 1j**jp*sp.comb(k, j)*kfac/(m2k-jp+1)
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism_y2(L, My, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the x-axis with a magnetization density Mx.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    My : float
        Horizontal magnetization density of the rectangular prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = -My*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m even, m odd -> l odd
    for l in range(1, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m odd
        for m in range(1, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = m+2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**k*H**(l-m2k+1)/2**(l+m2k-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        if (jp % 2) == 1:
                            kfac = (jp+1)*a**(m2k-jp+1)*b**(jp)
                        else:
                            kfac = (-1j)**m*(m2k-jp)*b**(m2k-jp)*a**(jp+1)
                        ksum += 1j**(jp)*sp.comb(k, j)*kfac/((m2k+1)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_prism_z(L, Mz, H, d, y1, y2):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the z-axis with a magnetization density Mz.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mz : float
        Vertical magnetization density of the triangular prism
    H : float
        Total height of the prism
    d : float
        Length of prism along x-axis
    y1 : float
    y2 : float

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mz*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (d <= 0) or (y2 <= y1):
        return qlm
    # l-m odd
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        # If l even then m odd and vice versa
        m0 = (l + 1) % 2
        for m in range(m0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m + 2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+1)
                slk = (-1)**(m+k)*H**(l-m2k)/2**(l-1)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        kfac = d**(m2k-jp+1)*(y2**(jp+1)-y1**(jp+1))
                        ksum += 1j**(jp)*sp.comb(k, j)*kfac/((m2k+2)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_prism_x(L, Mx, H, d, y1, y2):
    """
    Triangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the z-axis with a magnetization density Mz.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mx : float
        Horizontal x-oriented magnetization density of the triangular prism
    H : float
        Total height of the prism
    d : float
        Length of prism along x-axis
    y1 : float
    y2 : float

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mx*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (d <= 0) or (y2 <= y1):
        return qlm
    # l-m even
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        # If l even then m even and similarly, l odd -> m odd
        m0 = l % 2
        for m in range(m0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m + 2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**(m+k)*H**(l-m2k+1)/2**(l)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        kfac = (m2k-jp)*d**(m2k-jp)*(y2**(jp+1)-y1**(jp+1))
                        ksum += 1j**(jp)*sp.comb(k, j)*kfac/((m2k+1)*(jp+1))
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_prism_y(L, My, H, d, y1, y2):
    """
    Triangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0,
    magnetized along the z-axis with a magnetization density Mz.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    My : float
        Horizontal y-oriented magnetization density of the triangular prism
    H : float
        Total height of the prism
    d : float
        Length of prism along x-axis
    y1 : float
    y2 : float

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = My*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (d <= 0) or (y2 <= y1):
        return qlm
    # l-m even
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        # If l even then m even and similarly, l odd -> m odd
        m0 = l % 2
        for m in range(m0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m + 2*k
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m2k+2)
                slk = (-1)**(m+k)*H**(l-m2k+1)/2**(l)
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m+k, p)
                    for j in range(k+1):
                        jp = j+p
                        kfac = d**(m2k-jp+1)*(y2**jp-y1**jp)
                        ksum += 1j**(jp)*sp.comb(k, j)*kfac/(m2k+1)
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def annulus_z(L, Mz, H, Ri, Ro, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mz : float
        Vertical magnetization density of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mz*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            # Make sure (l-m) odd
            if ((l-m) % 2 == 1):
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+1)
                    slk = (-1)**(k+m)*H**(l-2*k-m)/(2**(l-2)*(2*k+m+2))
                    slk *= (Ro**(2*k+m+2) - Ri**(2*k+m+2))/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def annulus_r(L, Mr, H, Ri, Ro, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mr : float
        Radial magnetization density of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mr*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            # Make sure (l-m) even
            if ((l-m) % 2 == 0):
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m+1)*(2*k+m)/(2**(l-1)*(2*k+m+1))
                    slk *= (Ro**(2*k+m+1) - Ri**(2*k+m+1))/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def annulus_p(L, Mp, H, Ri, Ro, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mp : float
        Phi-oriented magnetization density of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = -Mp*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(1, l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= 1j*np.sin(m*phih)*np.exp(-1j*m*phic)
            # Make sure (l-m) even
            if ((l-m) % 2 == 0):
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m+1)/(2**(l-1)*(2*k+m+1))
                    slk *= (Ro**(2*k+m+1) - Ri**(2*k+m+1))/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def annulus_x(L, Mx, H, Ri, Ro, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mx : float
        x-oriented magnetization density of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mx*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            # Make sure (l-m) even
            if ((l-m) % 2 == 0):
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m+1)/(2**l*(2*k+m+1))
                    slk *= (Ro**(2*k+m+1) - Ri**(2*k+m+1))/np.exp(gamsum)
                    qlm[l, L+m] += slk*(2*k)*np.sinc((m+1)*phih/np.pi)
                    qlm[l, L+m] += slk*(2*k+2*m)*np.sinc((m-1)*phih/np.pi)
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def annulus_y(L, My, H, Ri, Ro, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    My : float
        y-oriented magnetization density of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = My*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= 1j*np.exp(-1j*m*phic)
            # Make sure (l-m) even
            if ((l-m) % 2 == 0):
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m+1)/(2**l*(2*k+m+1))
                    slk *= (Ro**(2*k+m+1) - Ri**(2*k+m+1))/np.exp(gamsum)
                    qlm[l, L+m] += slk*(2*k)*np.sinc((m+1)*phih/np.pi)
                    qlm[l, L+m] -= slk*(2*k+2*m)*np.sinc((m-1)*phih/np.pi)
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone_z(L, Mz, H, R, phic, phih):
    """
    We use the notation of Stirling and Schlamminger to compute the inner
    moments of a section of a cone. The solid has a radius R and extends above
    the xy-plane by H.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mz : float
        Vertical magnetization density of the cone section
    H : float
        Total height of the cone section
    R : float
        Radius of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mz*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R <= 0) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih*2
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            # Make sure (l-m-2k) > 0
            for k in range((l-m)//2+1):
                if ((l-m-2*k) > 0):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l+3)
                    gamsum -= sp.gammaln(2*k+m+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)/(2**(2*k+m))
                    slk *= R**(2*k+m+2)/np.exp(gamsum)
                    qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone_r(L, Mr, H, R, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of a section of a cone. This is a non-recursive
    attempt. The solid has a height H and extends above the xy-plane by H.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mr : float
        Radial magnetization density of the cone section
    H : float
        Total height of the cone section
    R : float
        Radius of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mr*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R <= 0) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih*2
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                if (m+2*k > 0):
                    m2k = 2*k+m
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l+3)
                    gamsum -= sp.gammaln(m2k+1)
                    slk = (-1)**(k+m)*H**(l-m2k+1)*m2k/(2**m2k)
                    slk *= R**(m2k+1)/np.exp(gamsum)
                    qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone_p(L, Mp, H, R, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of a section of a cone. This is a non-recursive
    attempt. The solid has a height H and extends above the xy-plane by H.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mp : float
        Phi-oriented magnetization density of the cone section
    H : float
        Total height of the cone section
    R : float
        Radius of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = -Mp*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R <= 0) or (phih == 0) or (phih > np.pi):
        return qlm
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(1, l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= 1j*np.sin(m*phih)*np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = 2*k+m
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l+3)
                gamsum -= sp.gammaln(m2k+1)
                slk = (-1)**(k+m)*H**(l-m2k+1)/(2**(m2k-1))
                slk *= R**(m2k+1)/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone_x(L, Mx, H, R, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of a section of a cone. This is a non-recursive
    attempt. The solid has a height H and extends above the xy-plane by H.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    Mx : float
        x-oriented magnetization density of the cone section
    H : float
        Total height of the cone section
    R : float
        Radius of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = Mx*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R <= 0) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = 2*k+m
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l+3)
                gamsum -= sp.gammaln(m2k+1)
                slk = (-1)**(k+m)*H**(l-m2k+1)/(2**(m2k-1))
                slk *= R**(m2k+1)/np.exp(gamsum)
                qlm[l, L+m] += slk*k*np.sinc((m+1)*phih/np.pi)
                qlm[l, L+m] += slk*(k+m)*np.sinc((m-1)*phih/np.pi)
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone_y(L, My, H, R, phic, phih):
    """
    Only L-M odd survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of a section of a cone. This is a non-recursive
    attempt. The solid has a height H and extends above the xy-plane by H.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    My : float
        y-oriented magnetization density of the cone section
    H : float
        Total height of the cone section
    R : float
        Radius of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = My*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R <= 0) or (phih == 0) or (phih > np.pi):
        return qlm
    factor *= phih
    for l in range(1, L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= 1j*np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                m2k = 2*k+m
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l+3)
                gamsum -= sp.gammaln(m2k+1)
                slk = (-1)**(k+m)*H**(l-m2k+1)/(2**(m2k-1))
                slk *= R**(m2k+1)/np.exp(gamsum)
                qlm[l, L+m] += slk*(k)*np.sinc((m+1)*phih/np.pi)
                qlm[l, L+m] -= slk*(k+m)*np.sinc((m-1)*phih/np.pi)
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm
