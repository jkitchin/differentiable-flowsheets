/* Loads difflow.gui's page script with just enough of a browser under
   it to exercise the pure model functions -- naming, wiring, seeding.
   Run by tests/test_gui.py, which extracts the script from _PAGE and
   passes it in PAGE_JS. The rendering is left to the browser; what is
   checked here is the bookkeeping that would silently disconnect a
   flowsheet if it were wrong.

   Nothing in difflow needs node. The test that drives this skips when
   node is not installed.
*/
const vm = require("vm");
const src = require("fs").readFileSync(process.env.PAGE_JS, "utf8");

const noElement = () => ({
  addEventListener() {}, querySelectorAll: () => [],
  set innerHTML(v) {}, get innerHTML() { return ""; },
  textContent: "", setAttribute() {},
});
globalThis.document = { getElementById: noElement };
globalThis.fetch = async () => ({ json: async () => ({ ok: true }) });

vm.runInThisContext(src + `
;globalThis.T = {renameStream, producers, streamNames, newUnit, problems,
                 seedValue, seedParams, uniqueUnitName, freeStream, esc,
                 setDOC: d => { DOC = d; }, setCAT: c => { CATALOG = c; }};
`);
