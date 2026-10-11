"""Stage 1 prep: bed inlets (F, T, P, W) at WAIT 480/500/530 C from the existing solver."""
import pickle
import sys
import time

import numpy as np

from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
from difflow_refinery.reforming.reactor import flows_of

out = {}
base = None
for wait in (500.0, 480.0, 530.0):
    t = time.time()
    p = ReformerParams().with_wait(wait + 273.15)
    res = CatalyticReformer(p).solve(lean_naphtha(), tear_initial=None if base is None else base.tear,
                                     tol=1e-11, max_iter=400)
    if base is None:
        base = res
    beds = []
    for k in range(3):
        s = res.streams[f"rx{k+1}_in"]
        beds.append(dict(F=np.asarray(flows_of(s)), T=float(s["T"]), P=float(s["P"]),
                         W=float(res.reactors[k]["W"])))
    out[wait] = beds
    print(wait, res.converged, res.iterations, time.time() - t, flush=True)
pickle.dump(out, open(sys.argv[1], "wb"))
