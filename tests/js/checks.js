/* Assertions over the editor's model functions. See harness.js. */
require(process.env.HARNESS);
const T = globalThis.T;
let fails = 0;
const eq = (label, got, want) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.log("FAIL " + label + "\n  got  " + a + "\n  want " + b); fails++; }
  else console.log("ok   " + label);
};

const doc = () => ({flowsheet: {
  species_order: ["water", "ethanol"],
  feeds: {feed: {F_water: 1.0, T: 350.0}},
  units: [
    {name:"reactor", operation:"CSTR", params:{}, inlets:["feed"], outlets:["rx"]},
    {name:"flash", operation:"Flash", params:{}, inlets:["rx"], outlets:["liq","vap"]},
  ],
  recycles: {},
}});
T.setCAT({
  Heater: {name:"Heater", ports:{inlets:["s"], n_inlets:1, n_outlets:1, variadic:false},
           parameters:[{name:"T_out", type:"jax.Array | float | None", required:false}]},
  Mixer:  {name:"Mixer", ports:{inlets:[], n_inlets:null, n_outlets:1, variadic:true},
           parameters:[{name:"species_order", type:"list[str]", required:true}]},
  PSAUnit:{name:"PSAUnit", ports:{inlets:["s"], n_inlets:1, n_outlets:2, variadic:false},
           parameters:[{name:"adsorbent", type:"str", required:true},
                       {name:"n_beds", type:"int", required:true},
                       {name:"elements", type:"list[str]", required:true},
                       {name:"tol", type:"float", required:false}]},
});

/* --- producers and stream names --- */
T.setDOC(doc());
eq("streams are feeds plus outlets", T.streamNames(), ["feed","rx","liq","vap"]);
eq("free stream is the last thing nothing reads", T.freeStream(), "vap");

/* --- renaming a stream travels to its consumer --- */
let d = doc(); T.setDOC(d);
eq("rename returns true", T.renameStream("rx", "crude"), true);
eq("outlet renamed", d.flowsheet.units[0].outlets, ["crude"]);
eq("consumer follows", d.flowsheet.units[1].inlets, ["crude"]);

/* --- renaming a feed keeps it a feed, and the consumer follows --- */
d = doc(); T.setDOC(d);
T.renameStream("feed", "charge");
eq("feed key renamed", Object.keys(d.flowsheet.feeds), ["charge"]);
eq("feed values kept", d.flowsheet.feeds.charge.F_water, 1.0);
eq("consumer of the feed follows", d.flowsheet.units[0].inlets, ["charge"]);

/* --- a rename onto an existing name is refused, and changes nothing --- */
d = doc(); T.setDOC(d);
eq("collision refused", T.renameStream("rx", "liq"), false);
eq("nothing moved", d.flowsheet.units[0].outlets, ["rx"]);
eq("empty name refused", T.renameStream("rx", ""), false);

/* --- a recycle follows the rename too --- */
d = doc(); d.flowsheet.recycles = {vap: "rx"}; T.setDOC(d);
T.renameStream("vap", "overhead");
eq("recycle source follows", d.flowsheet.recycles, {overhead: "rx"});
d = doc(); d.flowsheet.recycles = {vap: "rx"}; T.setDOC(d);
T.renameStream("rx", "crude");
eq("recycle destination follows", d.flowsheet.recycles, {vap: "crude"});
eq("a recycle destination counts as produced",
   T.streamNames().includes("crude"), true);

/* --- adding a unit --- */
d = doc(); T.setDOC(d);
eq("new unit is wired to the free stream", T.newUnit("Heater"),
   {name:"heater", operation:"Heater", params:{}, constructor:{},
    extra_params:{}, inlets:["vap"], outlets:["heater_out"]});
d.flowsheet.units.push(T.newUnit("Heater"));
eq("second one gets a distinct name", T.newUnit("Heater").name, "heater2");
eq("and its own outlet name", T.newUnit("Heater").outlets, ["heater2_out"]);

d = doc(); T.setDOC(d);
eq("two outlets are numbered", T.newUnit("PSAUnit").outlets,
   ["psaunit_out1", "psaunit_out2"]);
eq("required params are seeded by type", T.newUnit("PSAUnit").params,
   {adsorbent: "", n_beds: 1.0, elements: []});
eq("a variadic unit starts with one inlet", T.newUnit("Mixer").inlets, ["vap"]);
eq("species_order is seeded from the flowsheet", T.newUnit("Mixer").params,
   {species_order: ["water", "ethanol"]});

/* --- problems --- */
d = doc(); T.setDOC(d);
eq("a sound flowsheet has no problems", T.problems(), []);
d = doc(); d.flowsheet.units[1].inlets = ["ghost"]; T.setDOC(d);
eq("a dangling inlet is named", T.problems(),
   ["flash reads ghost, which nothing produces"]);
d = doc(); d.flowsheet.units[1].inlets = [""]; T.setDOC(d);
eq("an unconnected inlet is named", T.problems(),
   ["flash has an inlet connected to nothing"]);
d = doc(); d.flowsheet.units[1].outlets = ["rx", "vap"]; T.setDOC(d);
eq("two producers of one stream is a problem", T.problems(),
   ["rx is produced by both unit reactor and unit flash"]);
d = doc(); d.flowsheet.units[1].name = "reactor"; T.setDOC(d);
eq("a duplicate unit name is a problem", T.problems()[0],
   "two units are named reactor");

/* --- escaping --- */
eq("attribute values are escaped", T.esc('a"b<c&d'), "a&quot;b&lt;c&amp;d");

console.log(fails ? "\n" + fails + " FAILED" : "\nall passed");
process.exit(fails ? 1 : 0);
