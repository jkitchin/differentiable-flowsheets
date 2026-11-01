"""n-hexane -> 2-methylpentane in a CSTR, 80 % of the effluent recycled."""
from difflow import (CSTR, CSTRParams, Flowsheet, IdealThermo, Mixer, Splitter, Unit,
                     make_stream, mass_action_kinetics)
from difflow.database import get_species_data

species = ["n_hexane", "2_methylpentane"]
thermo = IdealThermo({s: get_species_data(s) for s in species})
kin = mass_action_kinetics(
    [{"reactants": {species[0]: 1.0}, "products": {species[1]: 1.0},
      "rate_params": {"A": 0.05}}], species_order=species)
reactor = CSTR(CSTRParams(V=1.0, molar_density=10.0, **kin.params_kwargs()), thermo)

fs = Flowsheet(species_order=species)
fs.add_feed("fresh", make_stream({species[0]: 1.0, species[1]: 0.0}, T=340.0, P=101325.0))
fs.add_unit(Unit("mixer", Mixer(species, thermo), ["fresh", "recycle"], ["reactor_feed"]))
fs.add_unit(Unit("reactor", reactor, ["reactor_feed"], ["effluent"], params={"T_spec": 340.0}))
fs.add_unit(Unit("split", Splitter(species), ["effluent"], ["recycle", "product"],
                 params={"split_frac": 0.8}))
fs.add_recycle("recycle", "recycle")
