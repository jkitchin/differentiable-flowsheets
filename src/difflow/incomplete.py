"""A unit that is on the flowsheet but not yet finished.

Half the catalog needs something a palette drop cannot invent: a rate
law is code, an adsorbent is a material name, a ``thermo`` is an object.
Forty-five of the eighty-seven registered operations are in that
position the moment they are dropped.

They used to be held off the flowsheet entirely until the missing thing
appeared --- which meant they had no ports, and a unit with no ports
cannot be wired to anything. Dropping a compressor and a reactor and
trying to connect them simply did not work, and nothing on the canvas
said why.

So the unit goes on the flowsheet immediately, with its real ports, and
this stands in for the operation it does not have yet. Wiring is a
stream rename in difflow, so every existing edit works on it unchanged.
What it will not do is run: calling it raises, and the editor refuses to
solve while one is present, which is the honest answer. Inventing a
plausible ``adsorbent`` instead would have produced real-looking numbers
from a material nobody chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class IncompleteUnitError(RuntimeError):
    """Raised when an unfinished unit is asked to do the thing it cannot."""


@dataclass
class Incomplete:
    """The operation a unit is going to have, once it can be built.

    Attributes:
        operation: the registered name that was dropped, e.g. ``"CSTR"``.
        needs: what is still missing, by parameter name.
        hint: the sentence the editor shows about how to supply it.
    """

    operation: str
    needs: list[str] = field(default_factory=list)
    hint: str = ""

    #: Nothing was chosen, so there is nothing to report or to edit.
    params: dict = field(default_factory=dict, init=False)

    def __call__(self, *args, **kwargs):
        missing = ", ".join(self.needs) or "something it cannot guess"
        raise IncompleteUnitError(
            f"{self.operation} is not finished: it still needs {missing}. "
            "Define it in the code context and the unit builds itself, "
            "keeping the wiring it already has."
        )
