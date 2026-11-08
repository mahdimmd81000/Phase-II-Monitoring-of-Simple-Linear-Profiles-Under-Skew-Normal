'''EWMA_3 comparison tables on the common manuscript design N=10, x=1..10.

This is the only EWMA_3 implementation retained for the manuscript comparison.
All three KMW-style streams are used: two-sided EWMA intercept/slope and the
one-sided lower-clamped log(MSE) EWMA variance chart. Errors are centred
skew-normal errors. Standard-deviation shifts use the nominal additive
convention sigma0(a) -> sigma0(a) + d*SIGMA, matching the corrected EWMA/R
comparison path.
'''
from __future__ import annotations
import argparse, json, time, zlib
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
SEED=20260825
N_OBS=10
X=np.arange(1., 11.)
XBAR=X.mean()
XC=X-XBAR
SXX=float(np.sum(XC**2))
A0=3.; A1=2.; SIGMA=1.; LAM=0.2
SHIFTS=[round(.2*i,1) for i in range(1,11)]
SLOPE_SHIFTS=[round(.025*i,3) for i in range(1,11)]
SD_LEVELS=[round(1.0+.2*i,1) for i in range(1,11)]
TYPES=("intercept","slope","variance")
Q=N_OBS-2
VAR_LOG_MSE=2/Q+2/Q**2+4/(3*Q**3)-16/(15*Q**5)
VAR_SHIFT="additive"
JOINT_TARGET_ARL0=205.0

def sigma0(a):
    d=a/np.sqrt(1+a*a)
    return float(SIGMA*np.sqrt(1-2*d**2/np.pi))

def gen_errors(rng,a,shape,scale=SIGMA):
    d=a/np.sqrt(1+a*a); size=tuple(shape)+(N_OBS,)
    x=d*np.abs(rng.standard_normal(size))+np.sqrt(1-d*d)*rng.standard_normal(size)
    return (x-d*np.sqrt(2/np.pi))*scale

def params(kind, d, a):
    '''Apply the manuscript parameter-shift grid.

    The displayed intercept and slope grids are raw model-parameter shifts:
    A0 -> A0 + d*sigma and A1 -> A1 + d*sigma. The statistics are then
    standardized with exact OLS standard errors in stats().
    '''
    if kind == "intercept":
        return A0 + d*SIGMA, A1, SIGMA
    if kind == "slope":
        return A0, A1 + d*SIGMA, SIGMA
    if kind == "variance":
        sig=sigma0(a)
        # d is the displayed SD level sigma=1.2,...,3.0; convert to the
        # additive nominal increment before generating the profile.
        increment=d-SIGMA
        return A0, A1, (sig + increment*SIGMA) / sig
    return A0, A1, SIGMA

def stats(errors, kind, d, a):
    b0, b1, scale = params(kind, d, a)
    y=b0+b1*X+errors
    ybar=y.mean(-1); b1hat=(XC*(y-ybar[...,None])).sum(-1)/SXX
    b0hat=ybar-b1hat*XBAR
    resid=y-b0hat[...,None]-b1hat[...,None]*X
    sig=sigma0(a)
    # Exact OLS standardisation for the uncoded x=1,...,10 design.  The
    # intercept variance is sigma^2*(1/N + xbar^2/Sxx); sqrt(N)/sigma is
    # only valid for a centred intercept.  Using sqrt(N) here inflated the
    # in-control SD of u0 to about 2.16 and forced L0 to an artificial ~6.4.
    sd_b0=sig*np.sqrt(1.0/N_OBS + XBAR**2/SXX)
    u0=(b0hat-A0)/sd_b0
    u1=(b1hat-A1)*np.sqrt(SXX)/sig
    mse=(resid**2).sum(-1)/Q
    w=np.log(mse)
    return u0,u1,w

def run_path(u0,u1,w,z,limits,which="joint"):
    '''Vectorized EWMA walk; stop updating paths after their first alarm.'''

    m,H=u0.shape; z0=z[:,0].copy(); z1=z[:,1].copy(); z2=z[:,2].copy()
    b=np.sqrt(LAM/(2-LAM)); l0=limits["L0"]*b; l1=limits["L1"]*b
    base=limits["base"]; ucl=limits["ucl"]
    first=np.full(m,np.inf); alive=np.ones(m,dtype=bool)
    for t in range(H):
        if not alive.any(): break
        if which in ("joint","int"): z0[alive]=LAM*u0[alive,t]+(1-LAM)*z0[alive]
        if which in ("joint","slope"): z1[alive]=LAM*u1[alive,t]+(1-LAM)*z1[alive]
        if which in ("joint","var"): z2[alive]=np.maximum(LAM*w[alive,t]+(1-LAM)*z2[alive],base)
        h=np.zeros(m,dtype=bool)
        if which in ("joint","int"): h[alive] |= np.abs(z0[alive])>l0
        if which in ("joint","slope"): h[alive] |= np.abs(z1[alive])>l1
        if which in ("joint","var"): h[alive] |= z2[alive]>ucl
        new=h & alive; first[new]=t+1; alive &= ~new
    z_end=np.stack([z0,z1,z2],axis=1)
    return first,z_end

def arl(a,kind,d,limits,state,reps,horizon,seed,which="joint",warmup=50):
    rng=np.random.default_rng(seed); z=np.zeros((reps,3)); z[:,2]=limits["base"]
    if state=="steady":
        # Generate an in-control warm-up and retain the resulting state only for
        # paths with no warm-up alarm; restart rejected paths.
        ready=np.zeros(reps,bool)
        while not ready.all():
            idx=np.flatnonzero(~ready); e=gen_errors(rng,a,(len(idx),warmup)); u0,u1,w=stats(e,"ic",0,a)
            zi=np.zeros((len(idx),3)); zi[:,2]=limits["base"]
            f,zi_end=run_path(u0,u1,w,zi,limits,"joint")
            ok=~np.isfinite(f); ready[idx[ok]]=True
            z[idx[ok],:] = zi_end[ok,:]
    f=np.full(reps,np.nan); done=0; chunk=250
    scale=params(kind,d,a)[2]
    while np.isnan(f).any() and done<horizon:
        n=min(chunk,horizon-done); idx=np.flatnonzero(np.isnan(f)); e=gen_errors(rng,a,(len(idx),n),scale=scale)
        u0,u1,w=stats(e,kind,d,a); ff,z_end=run_path(u0,u1,w,z[idx],limits,which)
        new=np.isfinite(ff); f[idx[new]]=ff[new]+done; z[idx,:]=z_end; done+=n
    cap=20*horizon; ext=done
    while np.isnan(f).any() and ext<cap:
        n=min(chunk,cap-ext); idx=np.flatnonzero(np.isnan(f)); e=gen_errors(rng,a,(len(idx),n),scale=scale)
        u0,u1,w=stats(e,kind,d,a); ff,z_end=run_path(u0,u1,w,z[idx],limits,which); new=np.isfinite(ff); f[idx[new]]=ff[new]+ext; z[idx,:]=z_end; ext+=n
    vals=np.where(np.isfinite(f),f,cap); return float(vals.mean()),float(vals.std(ddof=1)/np.sqrt(reps))

def calibrate(a,reps,horizon,seed):
    sig=sigma0(a); base=np.log(sig**2); limits={"L0":3.,"L1":3.,"base":base,"ucl":base+1.}
    for key,which in (("L0","int"),("L1","slope"),("LE","var")):
        lo,hi=(1.,12.) if which != "var" else (.2,5.); best=None
        for it in range(28):
            mid=(lo+hi)/2
            if which=="var": limits["ucl"]=base+mid*np.sqrt(LAM/(2-LAM)*VAR_LOG_MSE)
            else: limits[key]=mid
            val,_,=arl(a,"ic",0,limits,"zero",reps,horizon,seed+it,which)
            best=(mid,val)
            if val<584: lo=mid
            else: hi=mid
            if abs(val-584)/584<.025: break
        if which=="var": limits["LE"]=best[0]
        else: limits[key]=best[0]
    limits["ucl"]=base+limits["LE"]*np.sqrt(LAM/(2-LAM)*VAR_LOG_MSE)

    # The marginal ARL0 targets are only an initial calibration.  Because the
    # three EWMA streams are monitored jointly, apply a second direct joint
    # calibration so the reported comparison chart has ARL0 close to 200.
    base_limits=dict(limits)
    lo_fac, hi_fac = 0.70, 1.60
    best=(1.0, float("inf"))
    for it in range(14):
        fac=(lo_fac+hi_fac)/2
        trial=dict(base_limits)
        trial["L0"]=base_limits["L0"]*fac
        trial["L1"]=base_limits["L1"]*fac
        trial["ucl"]=base_limits["base"] + (base_limits["ucl"]-base_limits["base"])*fac
        joint,_,=arl(a,"ic",0,trial,"zero",reps,horizon,seed+1000+it)
        if abs(joint-JOINT_TARGET_ARL0)<abs(best[1]-JOINT_TARGET_ARL0): best=(fac,joint)
        if joint < JOINT_TARGET_ARL0: lo_fac=fac
        else: hi_fac=fac
    fac=best[0]
    limits["L0"]=base_limits["L0"]*fac
    limits["L1"]=base_limits["L1"]*fac
    limits["ucl"]=base_limits["base"] + (base_limits["ucl"]-base_limits["base"])*fac
    limits["joint_calibration_factor"]=fac
    limits["joint_calibration_arl0"]=best[1]
    return limits

def md(rows,a,limits,state):
    ic=next(r for r in rows if r["shift_type"]=="none")
    lines=[f"# EWMA_3 common design N=10 (skewness={a:g}, {state}-state)","",f"- Design: y = 3 + 2x + e, x=1..10, N=10, lambda={LAM}; sigma0={sigma0(a):.6f}.",f"- Limits: L0={limits['L0']:.4f}, L1={limits['L1']:.4f}, LE={limits['LE']:.4f}, UCL_E={limits['ucl']:.6f}; direct joint calibration factor={limits.get('joint_calibration_factor',1.0):.4f}, calibrated joint ARL0={limits.get('joint_calibration_arl0',float('nan')):.2f}; nominal additive SD shift sigma0(a)->sigma0(a)+d.",f"- Joint IC ARL = {ic['arl']:.2f} (MCSE {ic['mcse']:.2f}).","| shift (intercept) / shift (slope) / sigma level | intercept | slope | SD |","|:--:|--:|--:|--:|"]
    lines.append(f"| 0.0 (IC) | {ic['arl']:.2f} | - | - |")
    def get(k,d): return next(r for r in rows if r["shift_type"]==k and abs(r["shift"]-d)<1e-9)
    for i,d in enumerate(SHIFTS):
        vals=[get("intercept",d),get("slope",SLOPE_SHIFTS[i]),get("variance",SD_LEVELS[i])]
        lines.append(f"| {d:.1f} / {SLOPE_SHIFTS[i]:.3f} / {SD_LEVELS[i]:.1f} | "+" | ".join(f"{v['arl']:.2f}" for v in vals)+" |")
    return "\n".join(lines)

def main():
    global JOINT_TARGET_ARL0
    ap=argparse.ArgumentParser(); ap.add_argument("--skewness",type=float,nargs="+",default=[0,.5,1,2]); ap.add_argument("--states",nargs="+",default=["zero","steady"]); ap.add_argument("--reps",type=int,default=2000); ap.add_argument("--horizon",type=int,default=1000); ap.add_argument("--cal-reps",type=int,default=1500); ap.add_argument("--joint-target",type=float,default=JOINT_TARGET_ARL0); ap.add_argument("--seed",type=int,default=SEED); x=ap.parse_args()
    JOINT_TARGET_ARL0=x.joint_target
    allrows=[]; cals={}
    for a in x.skewness:
        for state in x.states:
            print(f"skewness={a:g}, state={state}",flush=True); lim=calibrate(a,x.cal_reps,max(1000, min(2000, x.horizon)),x.seed+int(10*a)); cals[f"{a:g}_{state}"]=lim; rows=[]
            v,se=arl(a,"ic",0,lim,state,x.reps,max(x.horizon,4000),x.seed+900); rows.append({"skewness":a,"shift_type":"none","shift":0.,"state":state,"arl":v,"mcse":se,"replications":x.reps})
            for k in TYPES:
                shifts=SLOPE_SHIFTS if k=="slope" else (SD_LEVELS if k=="variance" else SHIFTS)
                for i,d in enumerate(shifts):
                    v,se=arl(a,k,d,lim,state,x.reps,x.horizon,x.seed+1500+i+zlib.crc32(f"{k}:{d}".encode())%100000)
                    rows.append({"skewness":a,"shift_type":k,"shift":d,"state":state,"arl":v,"mcse":se,"replications":x.reps,"sd_level":d if k=="variance" else np.nan})
            allrows+=rows; (BASE/f"ewma3_n10_table_skewness_{a:g}_{state}.md").write_text(md(rows,a,lim,state),encoding="utf-8")
    pd.DataFrame(allrows).to_csv(BASE/"ewma3_n10_results.csv",index=False); (BASE/"ewma3_n10_calibration.json").write_text(json.dumps(cals,indent=2),encoding="utf-8")
if __name__=="__main__": main()
