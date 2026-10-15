"""Check the CCR reformer's implicit derivative against central differences (#474).

Run from the repository root (about 10 minutes on a laptop)::

    PYTHONPATH=src:. python tests/reference/ccr_gradient_check.py

Not part of the pytest suite: ``jax.jacfwd`` through the CCR runs three
reformer solves in forward mode (the converged point, ``dPhi/dC`` by
``jacfwd``, the traced map), ~400 s, past ``DIFFLOW_TEST_BUDGET``.

It differentiates spent coke, reformate RON and peak burn T in
(circulation, WAIT) at the default design point of ``lean_naphtha()`` and
compares each column with a central difference (h = 1e-3 circulation,
0.05 K). Last run (2026-10-15): agreement to 3e-6 relative; the peak-T
row is zero (the peak is set by y_O2 alone).
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery.reforming import lean_naphtha
from difflow_refinery.reforming.ccr import CCRReformer

jax.config.update("jax_enable_x64", True)


def main(rtol: float = 2e-4) -> None:
    base = CCRReformer().solve(lean_naphtha())
    feed, u, p0 = lean_naphtha(), CCRReformer(base.params, base.ccr), base.params

    def f(x):
        r = u.solve(feed, p0.with_wait(x[1]), replace(base.ccr, circulation=x[0]),
                    tear_initial=base.tear)
        o = r.outputs()
        return jnp.stack([o["ccr.spent_coke_wt"], o["reformate.RON"], o["ccr.T_peak_C"]])

    x0 = jnp.array([float(base.ccr.circulation), float(p0.wait)])
    J = np.asarray(jax.jacfwd(f)(x0))
    for i, h in enumerate((1e-3 * float(x0[0]), 0.05)):
        e = jnp.zeros(2).at[i].set(h)
        fd = np.asarray((f(x0 + e) - f(x0 - e)) / (2 * h))
        print(f"column {i}: jacfwd {J[:, i]}  central FD {fd}")
        np.testing.assert_allclose(J[:2, i], fd[:2], rtol=rtol)
    np.testing.assert_allclose(J[2], 0.0, atol=1e-9)
    print("CCR jacfwd matches central differences")


if __name__ == "__main__":
    main()
