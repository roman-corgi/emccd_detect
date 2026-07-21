# -*- coding: utf-8 -*-
"""Example script for EMCCDDetect calls."""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect
try:
    import arcticpy as ap
except:
    pass
from emccd_detect.partial_CIC_MLE import EM_gain_fit_conv, EM_gain_fit_conv_rn, EM_gain_fit_LPG_W, EM_gain_fit_W


def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
    """Plot a scaled colormap."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect,
                   origin='lower')

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == '__main__':
    # Set up some inputs here
    here = os.path.abspath(os.path.dirname(__file__))
    from cal.util.gsw_process import Process
    from cal.util.read_metadata import Metadata as MetadataWrapper
    import cal

    libfile = cal.__path__[0]
    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(libfile, 'util', 'metadata.yaml')
    # no actual nonlinearity correction assumed here
    nonlin_path = Path(libfile, 'util', 'testdata', 'ut_nonlin_array_ones.txt')
    meta = MetadataWrapper(meta_path)
    image_rows, image_cols, r0c0 = meta._unpack_geom('image')
    
    np.random.seed(123)

    # full-well capacities
    fwc_em_e = 105000 #e-, for EM gain register
    fwc_pp_e = 90000 #e-, per pixel (before EM gain register)
    frametime = 1

    def read_in_files(directory, eperdn, bias_offset, gain, prescan=False):
        '''This function mainly subtracts the bias and 
        converts from DN to e-.  See Process class methods for details 
        on the processing done for these specific EMCCD frames that are used
        for demonstration in this script. 
        '''
        proc = Process(bad_pix=np.zeros((meta.frame_rows,meta.frame_cols)), eperdn=eperdn,
                                fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                                bias_offset=bias_offset, em_gain=gain,
                                exptime=frametime, nonlin_path=nonlin_path,
                                meta_path=meta_path)
        framelist = []
        for file in os.listdir(directory):
            if file.endswith('fits'):
                f = os.path.join(directory, file)
                d = fits.getdata(f)
                _, _, _, _, f0, b0, _ = proc.L1_to_L2a(d)
                # could just skip L2a_to_L2b(), but I like having a hook for
                # combining b0 with a proc_dark.bad_pix that isn't all zeros
                f1, b1, _ = proc.L2a_to_L2b(f0, b0)
                # to undo the division by gain in L2a_to_L2b()
                ff = f1*gain
                # assign NaN values to any pixels marked as bad (i.e., due to cosmic rays)
                ff = np.ma.masked_array(ff, mask=b1.astype(bool))
                f1 = ff.astype(float).filled(np.nan)
                # assign NaN values to last row, which does not have any physical readout
                f1[-1] = np.nan

                if prescan:
                    f1 = proc.meta.slice_section(f1, 'prescan')
                else:
                    f1 = proc.meta.slice_section(f1, 'image')
                framelist.append(f1)
        st = np.stack(framelist)
        return st

    def read_in_frame(frame, eperdn, bias_offset, gain, prescan=False):
        '''This function mainly subtracts the bias and 
        converts from DN to e-.  See Process class methods for details 
        on the processing done for these specific EMCCD frames that are used
        for demonstration in this script. 
        '''
        proc = Process(bad_pix=np.zeros((meta.frame_rows,meta.frame_cols)), eperdn=eperdn,
                                fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                                bias_offset=bias_offset, em_gain=gain,
                                exptime=frametime, nonlin_path=nonlin_path,
                                meta_path=meta_path)
        
        d = frame
        _, _, _, _, f0, b0, _ = proc.L1_to_L2a(d)
        # could just skip L2a_to_L2b(), but I like having a hook for
        # combining b0 with a proc_dark.bad_pix that isn't all zeros
        f1, b1, _ = proc.L2a_to_L2b(f0, b0)
        # to undo the division by gain in L2a_to_L2b()
        ff = f1*gain
        # assign NaN values to any pixels marked as bad (i.e., due to cosmic rays)
        ff = np.ma.masked_array(ff, mask=b1.astype(bool))
        f1 = ff.astype(float).filled(np.nan)
        # assign NaN values to last row, which does not have any physical readout
        f1[-1] = np.nan

        if prescan:
            f1 = proc.meta.slice_section(f1, 'prescan')
        else:
            f1 = proc.meta.slice_section(f1, 'image')
        return f1

    meta_path = Path(here, 'emccd_detect', 'util', 'metadata.yaml') 

    gain_CIC_specs = {}
    for r in range(200,400): #range(1,400):
        gain_CIC_specs[r] = .001
    em_gain = 5000 #1000 #5000.
    numel_gain_register=604
    gain_P = em_gain**(1/numel_gain_register) - 1
    emccd = EMCCDDetect(
        em_gain=em_gain,
        full_well_image=90000.,  # e-
        full_well_serial=105000.,  # e-
        dark_current=0.001,  # e-/pix/s
        cic=0.016,  # e-/pix/frame
        read_noise=110., # e-/pix/frame
        bias=1500.,  # e-
        qe=0.9,
        cr_rate=0, #5 # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=8.2,
        nbits=14,
        numel_gain_register=604,
        meta_path=meta_path,
        row_read_time=223.5e-6, # in seconds
        upstream_spill_prob=0.7,
        fast_gain_mode=False,
        gain_CIC_Q='roman',#gain_P/100, #0.001, #0.001, #0.001, #0.001, #0
        gain_CIC_specs= None, #{i:gain_P*5/(i) for i in range(1, 605)},#{i:gain_P/10 for i in range(1, numel_gain_register+ 1)}, #{i:gain_P/10 for i in range(int(np.round(numel_gain_register/4)), int(np.round(3*numel_gain_register/4) + 1))}, #gain_CIC_specs, #{200:.01,204:.01,300:.01,400:.01}, #gain_CIC_specs, #None,
        gain_stage_specs=None
    )


    # dark frame
    full_fluxmap = np.zeros((1024, 1024)).astype(float)
    # Specify frametime
    frametime = 1 # s
    import time 
    t = time.time()
    sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
    print("time for first frame: ", time.time() -t )
    directory = os.path.join(here, 'gain_stats_comparison')
    if not os.path.exists(directory):
        os.mkdir(directory)
    fits.writeto(os.path.join(directory, 'sim_dark.fits'), sim_full_frame, overwrite=True)

    frames = read_in_files(directory, eperdn=8.2, bias_offset=0, gain=5000)
    pass

    plots = True
    if plots:
        # emccd.gain_CIC_specs = None#{100:0.02,300:.003} #{i:.001 for i in range(200,401)}#{100: .02,300:.003}
        # emccd.gain_stage_specs = {300:0.2} #{44:.1} 
        emccd.gain_CIC_Q = 0
        # dark frame
        full_fluxmap = np.zeros((1024, 1024)).astype(float)
        # Specify frametime
        frametime = 1 # s
        np.random.seed(123) #same seed used above
        t = time.time()
        sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
        print("time for second frame: ", time.time() -t )
        sim_full_f = read_in_frame(sim_full_frame, eperdn=8.2, bias_offset=0, gain=5000)

        if frames.size < 1E6:
            title = '1 row, commanded gain of 5000'
        else:
            title = '1 frame, commanded gain of 5000'
        
        bi = int(np.round((frames.max()-frames.min())))
        y_vals, bin_edges = np.histogram(frames, bins=bi)
        x_vals = bin_edges[:-1]
        y_vals2, _ = np.histogram(sim_full_f, bins=bin_edges)

        fig, (ax, ax_resid) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 8),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3}
        )
        ax.tick_params(labeltop=False, labelbottom=True)
        ax.plot(x_vals, y_vals, label=r'normal stages')#r'Q=0.001 for stages 200-400')
        ax.plot(x_vals, y_vals2, label=r'P=0.2 for stage 300')#'Q=0.02 for stage 100, 0.003 for stage 300')
        ax.set_xlim(-1000, 2500)
        ax.set_ylabel('Count frequency')
        ax.set_title(title)
        ax.legend()

        ax_resid.plot(x_vals, y_vals - y_vals2, color='black', label='Residual (blue - orange)')
        ax_resid.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_resid.set_xlim(-1000, 2500)
        ax_resid.set_xlabel('Electron counts')
        ax_resid.set_ylabel('Residual')
        ax_resid.set_title('Residual difference between blue and orange plots')
        ax_resid.legend(fontsize='small')

        # Create inset of the log plot
        ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.20])
        ax_inset.semilogy(x_vals, y_vals, label='No partial CIC')
        ax_inset.semilogy(x_vals, y_vals2, label='Q=0.001 for stages 200-400')
        ax_inset.set_xlabel('Electron counts')
        ax_inset.set_ylabel('Count frequency')
        ax_inset.set_title('Logarithmic Scale')
        ax_inset.legend(fontsize='small')

        plt.show()

        
    res1, chisquare_value1, pvalue1  = EM_gain_fit_conv(frames,.008, 5000,6000,110,0,cut=-800, tol=1e-10)
    print(res1.success, res1.x, res1.fun, chisquare_value1, pvalue1)
    # 1. shown below:
    # for r in range(200,400):
    #     gain_CIC_specs[r] = .001
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    #True [2.80163944e-02 4.99998085e+03] 6696546.165994482 0.11868517551176494 1.0
    # for {200:.01,204:.01,300:.01,400:.01}:
    #True [1.98706939e-02 4.99999681e+03] 6594222.480285259 0.10855622575645495 1.0

    # No partial CIC, lambda, g, sigma_rn, and mu (Fits 2 and 6 from paper)
    res2, chisquare_value2, pvalue2 = EM_gain_fit_conv_rn(frames,.008,5000,6000,110,0,lthresh=0,cut=-800, tol=1e-10)
    print(res2.success, res2.x, res2.fun, chisquare_value2, pvalue2)
    # 2. shown below:
    # for r in range(200,400):
    #     gain_CIC_specs[r] = .001
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # True [ 2.61156820e-02  4.99986922e+03  1.18931378e+02 -1.83518547e+00] 6690626.255428597 0.11205443832476113 1.0
    # for {200:.01,204:.01,300:.01,400:.01}:
    #True [ 1.98439641e-02  4.99997982e+03  1.11893600e+02 -3.50573273e+00] 6593399.331218548 0.10793991757310924 1.0

    # With partial CIC, lambda, Q, and g (Fits 3 and 7 from paper)
    res3, chisquare_value3, pvalue3 = EM_gain_fit_LPG_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)#, Pn=True)
    print(res3.success, res3.x, res3.fun, chisquare_value3, pvalue3)
    # for 0.001 for all gain stages:
    #True [1.40963294e-01 1.42012057e-02 2.61188950e+03] 7403657.284383533 0.1860058273092654 1.0
    # 3. shown below for    
    # for r in range(200,400):
    #     gain_CIC_specs[r] = .001
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # (True [2.28108627e-02 7.09287935e-04 5.00001292e+03] 6692809.966892999 0.11009453319826594 1.0

    # slightly better using Gamma for _W:
    #True [2.27765809e-02 7.14602648e-04 5.00000367e+03] 6692788.182959802 0.10775965869086811 1.0

    # likelihood ratio test b/w this 3. and 2.:  -2*(-6692809.966892999 - (-6690626.255428597)) = 4367.422928802669 > 10.83 --> 2. (with 1 more parameter) statistically favored. Used Pn=True for 3. here.

    # for {200:.01,204:.01,300:.01,400:.001}:
    #True [1.74898725e-02 0.00000000e+00 4.99999792e+03] 6571691.174288297 0.10383917478013696 1.0
    # for {200:.01,204:.01,300:.01,400:.01}:
    #True [1.98707018e-02 0.00000000e+00 4.99999799e+03] 6594221.560174669 0.10855622250941128 1.0
    # for 0.001 for all stages (binomial in partial CIC):
    #True [1.67905292e-01 1.42012057e-02 2.63007034e+03] 7405219.246697019 0.16590738385130144 1.0
    # for 0.001 for all stages (poisson in partial CIC):
    #True [1.70135193e-01 0.00000000e+00 4.99878259e+03] 7478745.899167471 0.15684759858372635 1.0
    # for P1 for gain=5000 and P2 = P1/10, using two binomial calls per loop to simulate a trinomial distribution
    #True [2.82011173e-01 0.00000000e+00 4.99823430e+03] 7999846.632968326 0.21010066166531274 1.0

    # With partial CIC, lambda, Q, g, sigma_rn, and mu (Fits 4 and 8 from paper)
    res4, chisquare_value4, pvalue4 = EM_gain_fit_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)
    print(res4.success, res4.x, res4.fun, chisquare_value4, pvalue4)
    # 4. shown below:
        # 3. shown below for    
    # for r in range(200,400):
    #     gain_CIC_specs[r] = .001
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # for Gamma for _W:
    # True [ 2.169e-02  8.541e-04  5.000e+03  1.175e+02 -7.512e+00] 6686580.8875652775 0.10243103312002105 1.0
    # # likelihood ratio test b/w this 2. and 4.: -2*(-6690626.255428597 - (-6686580.8875652775)) = 8090.735726639628 > 10.83. --> 4. statistically favored (4. used Gamma for _W).

    # for {200:.01,204:.01,300:.01,400:.001}:
    #True [ 1.75748078e-02  0.00000000e+00  4.99999520e+03  1.09230741e+02 -1.41644281e+00] 

    # for {200:.01,204:.01,300:.01,400:.01}:
    ## for Pn version: True [ 1.97180745e-02  8.04675866e-05  4.99996782e+03  1.12367489e+02 -2.36753377e+00] 6594675.3457188085 0.11560425230391143 1.0
    ## most recent:  True [ 1.97210031e-02  1.15192129e-05  4.99999208e+03  1.12472438e+02 -2.47251071e+00] 6593464.5868936125 0.10789521119792946 1.0
    
    # for P1 for gain=5000 and P2 = P1/10, using two binomial calls per loop to simulate a trinomial distribution
    #True [ 2.83886731e-01  0.00000000e+00  4.78905214e+03  1.29479166e+02 -3.29298153e+01] 7903585.143631439 0.1632233171500318 1.0

    # for real data, gain~ 10, illuminated, one frame:
    #'/Users/kevinludwick/Documents/G 10 HV 25_0 DC 3 V Light 051821 CCD 85 EDU_ACQUIRE_NIMO_CB_JPLPS_051721_SN003/subfolder'
    # used read_in_files(), then frames = frames[0,250:650,200:600]
    # inputs to use:
    #res1, chisquare_value1, pvalue1  = EM_gain_fit_conv(frames, frames.mean()/10, 10,6000,110,0, tol=1e-10)
    #print(res1.success, res1.x, res1.fun, chisquare_value1, pvalue1)
    #True [881.23137529  15.        ] 1281667.317067626 0.09515188827345736 1.0
    
    #res2, chisquare_value2, pvalue2 = EM_gain_fit_conv_rn(frames, frames.mean()/10, 10,6000,110,0, tol=1e-10)
    #print(res2.success, res2.x, res2.fun, chisquare_value2, pvalue2)
    #True [1056.90322698   12.49467454  110.            0.        ] 1287831.3790719807 0.16097231059858086 1.0