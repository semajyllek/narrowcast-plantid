"""Does the 17.9 MB encoder work at the K a user actually picks?

Every encoder comparison in this project is at K=490. SMALL_FRONTIER_FINDINGS
notes "s2 at K=20 is a different story" and stops. This measures that story:
species top-1 and label-level share at the K a garden app builds, on the same
label sets, through the unmodified cascade.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import analysis.headroom_arms as H

ENCS=["bioclip2","plantclef24","mobileclip2_s2_ft","mobileclip2_s2"]
MB={"bioclip2":152,"plantclef24":43,"mobileclip2_s2_ft":17.9,"mobileclip2_s2":17.9}
P_OOD=0.2; rows=[]
ref,_=H.load("bioclip2")
allsp=np.array(sorted(set(ref["leaf"][1])|set(ref["flower"][1])))
for enc in ENCS:
    try: cat,bg=H.load(enc)
    except Exception as e: print(f"skip {enc}: {e}",flush=True); continue
    for K in (10,20,30):
        for crowded in (False,True):
            sets=H.draw_label_sets(allsp,K,3,np.random.default_rng(K+7*crowded),crowded)
            for si,sp in enumerate(sets):
                arm=H.fit_arm(cat,bg,sp,np.random.default_rng(si))
                gmap={s:s.split()[0] for s in sp}
                r=H.score(H.frame(arm,gmap),
                          dict(encoder=enc,mb=MB[enc],K=K,crowded=crowded,
                               set_id=f"K{K}-{'c' if crowded else 'v'}-{si}"),
                          p_ood=P_OOD)
                if r: rows.append(r)
        print(f"  {enc:20s} K={K} done",flush=True)
d=pd.DataFrame(rows); d.to_csv('/tmp/tiny_k.csv',index=False)
print("\n=== species top-1 (closed-set), by encoder and K ===")
print(d.pivot_table(index=['encoder','mb'],columns=['K','crowded'],values='fine').round(3).to_string())
print("\n=== label-level share: how often it NAMES a species ===")
print(d.pivot_table(index=['encoder','mb'],columns=['K','crowded'],values='label_share').round(3).to_string())
print("\n=== VARIED sets only: the product a user should be steered to ===")
v=d[~d.crowded]
print(v.pivot_table(index=['encoder','mb'],columns='K',values=['fine','label_share','coverage']).round(3).to_string())
