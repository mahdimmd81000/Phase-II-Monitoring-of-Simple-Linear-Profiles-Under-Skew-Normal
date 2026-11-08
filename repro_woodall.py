# -*- coding: utf-8 -*-
"""Reproduce Woodall's published Tables 1-3 with the EXACT KMW conventions.

Two corrections relative to the earlier tables:
  1. INTERCEPT: Woodall shifts the CENTRED intercept, i.e. the model is
     y = B0 + B1*(x - xbar) + e and the shift is B0 -> B0 + d*sigma.
     In the uncoded statistic u0 = (b0_hat - A0)/(sigma*sqrt(1/n+xbar^2/Sxx))
     this is a shift of d*sqrt(n) standard deviations, NOT d/sqrt(1/n+xbar^2/Sxx).
  2. SLOPE: shift b1 -> b1 + d*sigma (identical in both codings).
  3. SD: sigma -> g*sigma with the MULTIPLICATIVE convention used by Woodall
     (the ratio sigma_shift/sigma = g), and the R chart and the log-MSE chart
     are calibrated on that ratio.
"""
import numpy as np

import woodall2003_tables as w

w.IC_CAL_REPS = 1500
w.IC_CAL_HORIZON = 4000


def intercept_shift_stat(a, d):
    """Magnitude passed to params() so that u0 moves by d*sqrt(n) sd."""
    return d * np.sqrt(w.N_OBS) * w.SD_B0


print("Reproducing Woodall Tables 1-3 at skewness 0, ARL0 = 200, zero state")
print("=" * 74)

lim3 = w.calibrate3(0.0, w.IC_CAL_REPS, w.IC_CAL_HORIZON, 1)
schr = w.calibrate_r(0.0, w.IC_CAL_REPS, w.IC_CAL_HORIZON, 2)
calt2 = w.calibrate_t2(0.0, w.IC_CAL_REPS, w.IC_CAL_HORIZON, 3)
print("calibrated: EWMA3 %.1f  EWMAR %.1f  T2 %.1f"
      % (lim3["joint_arl0"], schr["joint_arl0"], calt2["joint_arl0"]), flush=True)

print("\nTable 1 (intercept, Woodall convention d*sqrt(n) sd):")
print("  d      EWMA3    EWMAR      T2")
pub1 = {0.2: (59.1, 66.5, 137.7), 0.4: (16.2, 17.7, 63.5),
        0.6: (7.9, 8.4, 28.0), 0.8: (5.1, 5.4, 13.2), 1.0: (3.8, 3.9, 6.9)}
for d in (0.2, 0.4, 0.6, 0.8, 1.0):
    mag = intercept_shift_stat(0.0, d)
    r3 = w.arl3(0.0, "intercept", mag, lim3, "zero", 1500, 700, 100 + int(d * 100))
    rr = w.arlr(0.0, "intercept", mag, schr, "zero", 1500, 700, 200 + int(d * 100))
    rt = w.arl_t2(0.0, "intercept", mag, calt2["ucl"], "zero", 3000, 700, 300 + int(d * 100))
    print("  %.1f   %6.1f   %6.1f   (published %.1f / %.1f / %.1f)"
          % (d, r3[0], rr[0], rt[0], pub1[d][0], pub1[d][1], pub1[d][2]))

print("\nTable 2 (slope, d*sigma):")
print("  d      EWMA3    EWMAR      T2")
pub2 = {0.025: (101.6, 119.0, 166.0), 0.05: (36.5, 43.9, 105.6),
        0.075: (17.0, 19.8, 60.7), 0.1: (10.3, 11.3, 34.5)}
for d in (0.025, 0.05, 0.075, 0.1):
    r3 = w.arl3(0.0, "slope", d, lim3, "zero", 1500, 700, 400 + int(d * 1000))
    rr = w.arlr(0.0, "slope", d, schr, "zero", 1500, 700, 500 + int(d * 1000))
    rt = w.arl_t2(0.0, "slope", d, calt2["ucl"], "zero", 3000, 700, 600 + int(d * 1000))
    print("  %.3f  %6.1f   %6.1f   (published %.1f / %.1f / %.1f)"
          % (d, r3[0], rr[0], rt[0], pub2[d][0], pub2[d][1], pub2[d][2]))

print("\nTable 3 (SD, multiplicative sigma -> g*sigma):")
print("  g      EWMA3    EWMAR      T2")
pub3 = {1.2: (33.5, 34.3, 39.6), 1.4: (12.7, 12.0, 14.9),
        1.6: (7.2, 6.1, 7.9), 1.8: (5.1, 3.9, 5.1)}
for g in (1.2, 1.4, 1.6, 1.8):
    r3 = w.arl3(0.0, "variance", g, lim3, "zero", 1500, 700, 700 + int(g * 100))
    rr = w.arlr(0.0, "variance", g, schr, "zero", 1500, 700, 800 + int(g * 100))
    rt = w.arl_t2(0.0, "variance", g, calt2["ucl"], "zero", 3000, 700, 900 + int(g * 100))
    print("  %.1f   %6.1f   %6.1f   (published %.1f / %.1f / %.1f)"
          % (g, r3[0], rr[0], rt[0], pub3[g][0], pub3[g][1], pub3[g][2]))