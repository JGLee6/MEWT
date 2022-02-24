#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:17:32 2022

@author: John Greendeer Lee
"""
import numpy as np
import newt.multipoleLib as mplb
import newt.translations as trs
import newt.rotations as rot
import mewt.mqlm as mqlm
import mewt.maglibShapes as mshp
import mewt.maglib as mglb

L = 10
IR = 0
OR = .05
H = .01
x_rad = .1
annz = mqlm.annulus_z(L, 1, H, IR, OR, 0, np.pi)
ds_z_x = rot.dlmn(L, np.pi/2)
annz_x = rot.rotate_qlm_Ds(annz, ds_z_x)
ds_z_x2 = rot.dlmn(L, -np.pi/2)
annz_x2 = rot.rotate_qlm_Ds(annz, ds_z_x2)
tgear2 = trs.translate_qlm(annz_x2, [x_rad, 0, 0])
tgear2 = rot.rotate_qlm(tgear2, np.pi/4, 0, 0)
tgear = trs.translate_qlm(annz_x, [x_rad, 0, 0])
gear = tgear + tgear2
Ds = rot.wignerDl(L, np.pi/2, 0, 0)
for k in range(3):
    tgear = rot.rotate_qlm_Ds(tgear, Ds)
    tgear2 = rot.rotate_qlm_Ds(tgear2, Ds)
    gear += tgear + tgear2

gear2 = np.copy(gear)
gear2 = trs.translate_q2Q(gear2, [4*x_rad, 0, 0])
tlm, tc, ts = mplb.torque_lm(L, gear, gear2)
ts *= mglb.magC/mplb.BIG_G

# Point-dipole version
annz = mshp.annulus(1, IR, OR, H, 1, 0, 0, 1, 4, 2)
annzx = mglb.rotate_dipole_array(annz, np.pi/2, [0, 1, 0])
annzx2 = mglb.rotate_dipole_array(annz, -np.pi/2, [0, 1, 0])
tgear = mglb.translate_dipole_array(annzx, [x_rad, 0, 0])
tgear2 = mglb.translate_dipole_array(annzx2, [x_rad, 0, 0])
tgear2 = mglb.rotate_dipole_array(tgear2, np.pi/4, [0, 0, 1])
gear = np.concatenate([tgear, tgear2])
np.shape(gear)
for k in range(3):
    gear = np.concatenate([gear, mglb.rotate_dipole_array(tgear, (k+1)*np.pi/2, [0, 0, 1])])
    gear = np.concatenate([gear, mglb.rotate_dipole_array(tgear2, (k+1)*np.pi/2, [0, 0, 1])])
    
gear2 = np.copy(gear)
gear2 = mglb.translate_dipole_array(gear2, [4*x_rad, 0, 0])
mglb.display_dipoles(gear, gear2, .01)

dmom = mglb.dmoments(L, gear)
Dmomb = mglb.Dmomentsb(L, gear2)
tlmb, tcb, tsb = mplb.torque_lm(L, dmom, Dmomb)
tsb *= mglb.magC/mplb.BIG_G
