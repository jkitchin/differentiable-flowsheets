// Guard for the docs assistant's retrieval half (_ext/ask_static/ask.js).
//
// Adapted from the POUNCE docs assistant's docs/tests/ask_retrieval.mjs. It
// loads the SHIPPED ask.js -- not a copy of the scoring -- through the test
// seam at the bottom of that file, and runs it against an index built from
// the rendered book. A test of a reimplemented ranker would stay green while
// the ranker readers actually use regressed, which is the whole failure mode
// here: retrieval has no exception to throw. It returns the wrong passage,
// silently, and the only symptom is a reader not finding the page.
//
// Two things are checked, and they fail for different reasons:
//
//   1. The stemmer, against known Porter step-1 outputs. A unit test: it does
//      not depend on what the book says.
//   2. Ranking, against a labelled query set. A *corpus* test, deliberately
//      loose -- see THRESHOLDS below.
//
// Usage (`make ask-check` does both steps; the deploy workflow runs the
// second after its own build):
//   jupyter-book build .
//   node tests/docs_ask/ask_retrieval.mjs _build/html/ask-index.json

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ask = require(resolve(here, "../../_ext/ask_static/ask.js"));

const indexPath = process.argv[2];
if (!indexPath) {
  console.error("usage: node tests/docs_ask/ask_retrieval.mjs <ask-index.json>");
  process.exit(2);
}

let failures = 0;
function check(ok, msg) {
  if (!ok) failures++;
  console.log((ok ? "  ok   " : "  FAIL ") + msg);
}

// ---------------------------------------------------------------- stemmer --

console.log("stemmer: pairs the corpus needs collapsed");
for (const [a, b] of [
  ["relaxing", "relax"],
  ["relaxed", "relax"],
  ["started", "start"],
  ["starting", "start"],
  // The +e restoration case: without it "scaling" stems to `scal` and stops
  // matching the page called "Scaling".
  ["scaling", "scale"],
  ["iterations", "iteration"],
  ["solves", "solve"],
  ["tightening", "tighten"],
  ["bounded", "bound"],
  ["duals", "dual"]
]) {
  check(ask.normTok(a) === ask.normTok(b), a + " ≡ " + b + " → " + ask.normTok(a));
}

console.log("stemmer: canonical Porter step-1 outputs");
for (const [w, want] of [
  ["caresses", "caress"],
  ["ponies", "poni"],
  ["cats", "cat"],
  ["feed", "feed"],
  ["agreed", "agree"],
  ["plastered", "plaster"],
  ["motoring", "motor"],
  ["hopping", "hop"],
  ["failing", "fail"],
  ["filing", "file"],
  ["happy", "happi"],
  ["sky", "sky"],
  // Known and accepted: Porter step 1 does not restore the `e` here, because
  // the cvc rule only fires at m == 1 and `converg` has m == 2. Pinned so the
  // asymmetry is a decision on the record rather than a surprise.
  ["converged", "converg"],
  ["converge", "converge"]
]) {
  check(ask.normTok(w) === want, w + " → " + ask.normTok(w));
}

console.log("stemmer: identifiers and short tokens pass through untouched");
for (const w of ["solve_eo", "molar_density", "tol", "lp", "co2", "h2o", "cvar", "jit"]) {
  check(ask.normTok(w) === w, w + " unchanged");
}

console.log("distinct terms must not collide");
for (const [a, b] of [
  ["scaling", "scalar"],
  ["convex", "converge"],
  ["dual", "duel"],
  ["bound", "bind"]
]) {
  check(ask.normTok(a) !== ask.normTok(b), a + " ≠ " + b);
}

// ---------------------------------------------------------------- ranking --

const doc = JSON.parse(readFileSync(indexPath, "utf8"));
const idx = ask.buildIndex(doc.chunks || []);
console.log("\nindex: " + idx.N + " passages, avg " + idx.avgdl.toFixed(0) + " tokens");
check(idx.N > 1000, "index is populated");
// Every kind of page the book carries must be in it: losing the notebooks
// would still answer, just worse, and nothing else here would notice.
for (const prefix of ["docs/", "examples/", "jax-tutorials/", "about-flowsheets/"]) {
  const n = (doc.chunks || []).filter((c) => c.u.startsWith(prefix)).length;
  check(n > 50, "passages from " + prefix + " present (" + n + ")");
}
check((doc.chunks || []).every((c) => !/^https?:|^\//.test(c.u)),
  "every citation is relative to the site root");
// The reason the index is built from the rendered HTML: a citation must land
// on a section Sphinx actually wrote, or it silently opens the page's top.
{
  const site = dirname(resolve(indexPath));
  const pages = new Map();
  let missing = [];
  for (const u of new Set((doc.chunks || []).map((c) => c.u))) {
    const [page, anchor] = u.split("#");
    if (!pages.has(page)) {
      let html = null;
      try { html = readFileSync(resolve(site, page), "utf8"); } catch (e) { /* missing */ }
      pages.set(page, html);
    }
    const html = pages.get(page);
    if (html === null || (anchor && !html.includes('id="' + anchor + '"'))) missing.push(u);
  }
  check(missing.length === 0, "every cited page and anchor exists in the build" +
    (missing.length ? " (missing: " + missing.slice(0, 5).join(", ") + ")" : ""));
}

console.log("\nquery-side compound handling");
{
  // An identifier the corpus contains is searched as itself, never split.
  const known = ask.queryTerms(idx, "solve_eo");
  check(known.length === 1 && known[0] === "solve_eo",
    "known identifier stays whole: [" + known.join(", ") + "]");
  const unknown = ask.queryTerms(idx, "no_such_option_here");
  check(unknown.length > 1, "unknown identifier falls back to parts: [" + unknown.join(", ") + "]");
  const stopped = ask.queryTerms(idx, "what does it do");
  check(stopped.length === 0, "an all-stopword question yields no terms");
}

console.log("\nresult shaping");
{
  const hits = ask.search(idx, "recycle", 6);
  const urls = hits.map((h) => h.chunk.u);
  check(new Set(urls).size === urls.length, "no two hits share a URL");
  const pages = urls.map((u) => u.split("#")[0]);
  const worst = Math.max(...pages.map((p) => pages.filter((q) => q === p).length));
  check(worst <= 2, "at most two hits per page (worst: " + worst + ")");
  check(ask.search(idx, "", 6).length === 0, "empty query returns nothing");
  check(ask.search(idx, "zzzqqqxxnotaword", 6).length === 0, "unmatched query returns nothing");
}

// Labelled set. `want` substrings are matched against the citation URL, and a
// query lists every page that genuinely answers it -- not one blessed page --
// because several of these questions have more than one honest home.
//
// THRESHOLDS ARE DELIBERATELY BELOW THE MEASURED SCORE. Adding or retitling a
// page can legitimately move a ranking, and a guard pinned to the current
// number would fail on honest doc edits and get raised until it meant
// nothing. It is set to catch a scoring change that drops several queries at
// once, not single-rank drift.
//
// Measured when the set was written (119 pages, 2841 passages): 23/25 top-1,
// 25/25 top-5. The two top-1 misses are honest: the gas notebook outranks the
// gas reference page on "Weymouth", and a Gaussian-process tutorial outranks
// experiment-design.md on "optimal experimental design".
const EVAL = [
  { q: "why does my recycle loop not converge",
    want: ["docs/convergence.html", "docs/streams-and-flowsheets.html"] },
  { q: "TearToleranceWarning", want: ["docs/convergence.html", "docs/streams-and-flowsheets.html"] },
  { q: "equation-oriented solver with Newton", want: ["docs/eo-solver.html", "docs/convergence.html"] },
  { q: "how do I install difflow", want: ["docs/getting-started.html", "README.html"] },
  { q: "Peng-Robinson equation of state", want: ["docs/thermodynamics.html", "examples/"] },
  { q: "CSTR molar density", want: ["docs/unit-operations-chemical.html"] },
  { q: "net present value and capital cost", want: ["docs/technoeconomics.html", "examples/"] },
  { q: "gross error detection in data reconciliation", want: ["docs/data-reconciliation.html"] },
  { q: "delta vectors and the trust region in planning", want: ["docs/planning.html", "examples/30"] },
  { q: "two-stage stochastic program with CVaR", want: ["docs/stochastic.html", "examples/32"] },
  { q: "AC optimal power flow locational marginal prices", want: ["docs/unit-operations-power.html"] },
  { q: "Weymouth equation for a gas pipe", want: ["docs/unit-operations-gas.html"] },
  { q: "rare earth solvent extraction distribution coefficient",
    want: ["docs/unit-operations-ree.html", "examples/"] },
  { q: "amine absorber for CO2 capture",
    want: ["docs/unit-operations-carbon-capture.html", "src/difflow_cc/examples/"] },
  { q: "fed-batch bioreactor", want: ["docs/unit-operations-bio.html", "examples/bio/"] },
  { q: "crude assay TBP characterization",
    want: ["docs/unit-operations-refinery.html", "docs/refinery-summary.html", "examples/"] },
  { q: "moving horizon estimation", want: ["docs/moving-horizon-estimation.html"] },
  { q: "parameter estimation confidence intervals",
    want: ["docs/parameter-estimation.html", "examples/"] },
  { q: "flexibility index of a design", want: ["docs/flexibility.html", "docs/operability.html"] },
  { q: "optimal experimental design", want: ["docs/experiment-design.html", "examples/"] },
  { q: "integrate a dynamic DAE model with diffrax", want: ["docs/dynamic-modeling.html", "examples/"] },
  { q: "what do jit and vmap do", want: ["jax-tutorials/"] },
  { q: "choosing tear streams automatically",
    want: ["docs/streams-and-flowsheets.html", "docs/convergence.html"] },
  // external-solvers.md never mentions Pyomo; planning.md's "Emitting Pyomo"
  // and notebook 30's hand-off are where the book answers this.
  { q: "solve with IPOPT through Pyomo", want: ["docs/planning.html#emitting-pyomo", "examples/30"] },
  { q: "export delta vectors to PIMS", want: ["docs/pims-integration.html", "docs/planning.html"] }
];
const THRESHOLD_TOP1 = 19;
const THRESHOLD_TOP5 = 22;

console.log("\nprompt construction (the RAG contract)");
{
  const hits = ask.search(idx, "solve_eo", 5);
  const msgs = ask.buildPrompt("what does solve_eo do?", hits);
  check(msgs.length === 2 && msgs[0].role === "system" && msgs[1].role === "user",
    "system + user message");
  // The grounding instructions are the only thing keeping a 1B model from
  // answering from memory; losing them would not fail loudly.
  check(/ONLY from the numbered excerpts/.test(msgs[0].content), "system prompt grounds the model");
  check(/Cite every claim/.test(msgs[0].content), "system prompt demands citations");
  check(/difflow/.test(msgs[0].content), "system prompt names difflow");
  check(msgs[1].content.includes("what does solve_eo do?"), "question is in the user message");
  for (let n = 1; n <= hits.length; n++) {
    check(msgs[1].content.includes("[" + n + "] "), "excerpt [" + n + "] is numbered");
  }
  const long = [{ chunk: { h: "H", t: "T", u: "u.html", x: "x".repeat(ask.MAX_CTX_CHARS * 3) } }];
  const cut = ask.buildPrompt("q", long)[1].content;
  check(cut.length < ask.MAX_CTX_CHARS * 2, "over-long passage is truncated (" + cut.length + " chars)");
  check(cut.includes("\u2026"), "truncation is marked with an ellipsis");
}

console.log("\nranking over " + EVAL.length + " labelled queries");
let top1 = 0;
let top5 = 0;
for (const c of EVAL) {
  const urls = ask.search(idx, c.q, 6).map((h) => h.chunk.u);
  const hit = (u) => c.want.some((w) => u.includes(w));
  const at1 = urls.length > 0 && hit(urls[0]);
  const at5 = urls.slice(0, 5).some(hit);
  if (at1) top1++;
  if (at5) top5++;
  console.log("  " + (at1 ? "T1" : at5 ? ".5" : "XX") + "  " + c.q);
  if (!at1) console.log("        got: " + (urls[0] || "(nothing)"));
}

console.log("\n  top-1 " + top1 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP1 + ")");
console.log("  top-5 " + top5 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP5 + ")");
check(top1 >= THRESHOLD_TOP1, "top-1 above the floor");
check(top5 >= THRESHOLD_TOP5, "top-5 above the floor");

console.log("\n" + (failures ? failures + " FAILURE(S)" : "ask_retrieval: OK"));
process.exit(failures ? 1 : 0);
