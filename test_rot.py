# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:27:33 2020

@author: jgl6
"""
import numpy as np
import maglib as mglb
import newt.translations as trs
import newt.rotations as rot
import newt.pg2Multi as pgm


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
# negative rotation about y-axis?
# z-dip at [0, 0, 1] -> rotate to x-dip at [1, 0, 0]
dmomxrt = rot.rotate_qlm(dmomzt, 0, np.pi/2, 0)
assert (np.abs(dmomxrt - dmomxt) < 4e2*np.finfo(float).eps).all()
# Fails for even l
# np.where(np.abs(dmomxrt - dmomxt) > 2e2*np.finfo(float).eps)

# What about for a positive rotation about y-axis?
# -x-dip at origin -> mom -> [-1, 0, 0]
magmx = np.array([[1, 0, 0, 0, 1, -1, 0, 0]])
dmommx = mglb.dmoments(lmax, magmx)
dmommxt = trs.translate_qlm(dmommx, [-1, 0, 0])
# -x-dip at [-1, 0, 0] -> mom
magmtx = np.array([[1, -1, 0, 0, 1, -1, 0, 0]])
dmommtx = mglb.dmoments(lmax, magmtx)
assert (np.abs(dmommtx - dmommxt) < 2e2*np.finfo(float).eps).all()
# z-dip at [0, 0, 1] -> rotate to -x-dip at [-1, 0, 0]
dmommxrt = rot.rotate_qlm(dmomzt, 0, -np.pi/2, 0)
assert (np.abs(dmommxrt - dmommxt) < 2e2*np.finfo(float).eps).all()
# Fails for even l still

# Does it fail just for gravity?
# z-dip at origin -> mom -> [0, 0, 1]
magz = np.array([[1, 0, 0, 0]])
dmomz = pgm.qmoments(lmax, magz)
dmomzt = trs.translate_qlm(dmomz, [0, 0, 1])
# z-dip at [0, 0, 1] -> mom
magtz = np.array([[1, 0, 0, 1]])
dmomtz = pgm.qmoments(lmax, magtz)
assert (np.abs(dmomtz - dmomzt) < 2e2*np.finfo(float).eps).all()
# x-dip at origin -> mom -> [1, 0, 0]
magx = np.array([[1, 0, 0, 0]])
dmomx = pgm.qmoments(lmax, magx)
dmomxt = trs.translate_qlm(dmomx, [1, 0, 0])
# x-dip at [1, 0, 0] -> mom
magtx = np.array([[1, 1, 0, 0]])
dmomtx = pgm.qmoments(lmax, magtx)
assert (np.abs(dmomtx - dmomxt) < 2e2*np.finfo(float).eps).all()
# negative rotation about y-axis?
# z-dip at [0, 0, 1] -> rotate to x-dip at [1, 0, 0]
dmomxrt = rot.rotate_qlm(dmomzt, 0, np.pi/2, 0)
assert (np.abs(dmomxrt - dmomxt) < 2e2*np.finfo(float).eps).all()
# Now all odd l!
np.where(np.abs(dmomxrt - dmomxt) > 2e2*np.finfo(float).eps)
