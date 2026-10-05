"""HOLD-only vector diagnostics: shared scales, seed uncertainty, no paper export."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR",str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root="runs/p1_x1"):
    root=Path(root)
    rows=json.loads((root/"rows.json").read_text())
    ns=sorted({r["n_cal"] for r in rows})
    ms=sorted({r["M"] for r in rows})
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"legend.fontsize":8,
                        "axes.spines.top":False,"axes.spines.right":False,
                        "pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,len(ns),figsize=(2.35*len(ns),2.7),sharex=True,sharey=True,squeeze=False)
    styles=[("pointwise_msce","Pointwise error","#0072B2","o","-"),
            ("approx","Within-cell component","#009E73","s","--"),
            ("msce","Cell-average error","#D55E00","^","-.")]
    for ax,n in zip(axes.flat,ns):
        for key,label,color,marker,line in styles:
            values=[np.array([r[key] for r in rows if r["n_cal"]==n and r["M"]==m]) for m in ms]
            means=np.array([v.mean() for v in values])
            # Across-seed SD, not an implied confidence interval.
            sd=np.array([v.std(ddof=1) if len(v)>1 else 0 for v in values])
            ax.plot(ms,means,color=color,marker=marker,linestyle=line,markersize=3,label=label,linewidth=1)
            ax.fill_between(ms,np.maximum(0,means-sd),means+sd,color=color,alpha=.12,linewidth=0)
        ax.set_xscale("log",base=2);ax.set_title(f"Calibration n = {n}")
        ax.set_xlabel("Number of cells M");ax.set_ylim(bottom=0)
    axes[0,0].set_ylabel("Squared coverage error")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.03),ncol=3,frameon=False)
    fig.text(.5,.015,"HOLD · diagnostic evidence only · bands: ±1 SD across seeds",ha="center",fontsize=8,color=".35")
    fig.tight_layout(rect=(0,.05,1,.92))
    dest=root/"figures";dest.mkdir(exist_ok=True)
    fig.savefig(dest/"decomposition.pdf",bbox_inches="tight")
    fig.savefig(dest/"decomposition.svg",bbox_inches="tight")
    fig.savefig(dest/"decomposition_preview.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.8))
    for n,color in zip(ns,["#0072B2","#D55E00","#009E73","#CC79A7","#000000"]):
        subset=[[r for r in rows if r["n_cal"]==n and r["M"]==m] for m in ms]
        axes[0].plot(ms,[np.mean([r["test_mass_infinite"] for r in v]) for v in subset],"o-",color=color,label=f"n={n}",markersize=3)
        axes[1].plot(ms,[np.mean([r["n_below_floor"] for r in v]) for v in subset],"o-",color=color,markersize=3)
    for ax in axes:
        ax.set_xscale("log",base=2);ax.set_xlabel("Number of cells M");ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Test mass with full prediction set")
    axes[1].set_ylabel("Calibration cells below floor")
    axes[0].legend(frameon=False)
    fig.suptitle("HOLD · floor diagnostics",fontsize=9)
    fig.tight_layout()
    fig.savefig(dest/"floor_diagnostics.pdf",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default="runs/p1_x1")
    plot(p.parse_args().root)
