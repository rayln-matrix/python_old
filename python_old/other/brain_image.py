# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:57:56 2020

@author: WB
"""


import os.path as op
import numpy as np
from surfer import io
from surfer import Brain


brain = Brain("fsaverage", "lh", "inflated", background="white",subjects_dir=r'C:\Users\WB\Desktop\Pysurfer')


#averagethick = io.read_scalar_data(op.join(r'C:\Users\WB\Desktop\Pysurfer', 'lh.curv.fsaverage.mgz'))
sig1 = io.read_scalar_data(op.join(r'C:\Users\WB\Desktop\Pysurfer', 'sch_lh_mc-z.abs.th13.sig.cluster.mgh'))
sig2 = io.read_scalar_data(op.join(r'C:\Users\WB\Desktop\Pysurfer', 'bd1_lh_mc-z.abs.th13.sig.cluster.mgh'))
sig3 = io.read_scalar_data(op.join(r'C:\Users\WB\Desktop\Pysurfer', 'bd2_lh_mc-z.abs.th13.sig.cluster.mgh'))
sig4 = io.read_scalar_data(op.join(r'C:\Users\WB\Desktop\Pysurfer', 'mdd_lh_mc-z.abs.th13.sig.cluster.mgh'))


thresh = 2
sig1[sig1 < thresh] = 0
sig2[sig2 < thresh] = 0
sig3[sig3 < thresh] = 0
sig4[sig4 < thresh] = 0

conjunct = np.min(np.vstack((sig1, sig2,sig3,sig4)), axis=0)



print(len(conjunct))

#brain.add_overlay(averagethick, 4, 30, name="avt")
#brain.overlays["avt"].pos_bar.lut_mode = "Blues"
#brain.overlays["avt"].pos_bar.visible = False

brain.add_overlay(sig2, 4, 30, name="sig1")
brain.overlays["sig1"].pos_bar.lut_mode = "Blues"
brain.overlays["sig1"].pos_bar.visible = False

brain.add_overlay(sig2, 4, 30, name="sig2")
brain.overlays["sig2"].pos_bar.lut_mode = "pink"
brain.overlays["sig2"].pos_bar.visible = False

brain.add_overlay(sig2, 4, 30, name="sig3")
brain.overlays["sig3"].pos_bar.lut_mode = "Greens"
brain.overlays["sig3"].pos_bar.visible = False

brain.add_overlay(sig2, 4, 30, name="sig4")
brain.overlays["sig4"].pos_bar.lut_mode = "Purples"
brain.overlays["sig4"].pos_bar.visible = False

'''brain.add_overlay(conjunct, 4, 30, name="conjunct")
brain.overlays["conjunct"].pos_bar.lut_mode = "Purples"
brain.overlays["conjunct"].pos_bar.visible = False'''