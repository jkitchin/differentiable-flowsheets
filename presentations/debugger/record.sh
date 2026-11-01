#!/usr/bin/env bash
# Re-record the deck's terminal sessions (real runs of `difflow debug` on
# demo/plant.py and demo/faulty.py) and redraw its figures. Each recording keeps
# its transcript (assets/NN-*.txt) and its last frame (assets/NN-*.png, the PDF's
# stand-in for the animation).
#
# The editor recordings (assets/panel-*.mp4) need the book's app (`difflow gui`)
# and node with playwright:
#   (cd demo && difflow gui plant.py --port 8801 --no-browser --stay --trust --token tok &)
#   curl -s -X POST -H X-Difflow-Token:tok -H Content-Type:application/json \
#        -d '{"acceleration":"wegstein"}' http://127.0.0.1:8801/api/solver
#   node record_gui.js "$PWD/video"     # then trim and encode, as below
#   ffmpeg -ss "$(cat video/debug.start)" -i video/debug.webm -c:v libx264 -crf 26 \
#          -pix_fmt yuv420p -movflags +faststart -an assets/panel-debug.mp4
set -euo pipefail
cd "$(dirname "$0")"
T="python record_term.py"
P="difflow debug plant.py --acceleration wegstein"

$T 01-start "$P" where next next "p reactor_feed" "p reactor" continue continue
$T 02-step "$P" step "p tear" step step "p history" continue continue
$T 03-breakpoints "$P" "break if residual<1e-6 && iter>1" "watch recycle.F_n_hexane" continue \
   "break clear" "break split" continue "p effluent" "break clear" continue continue
$T 04-on-error "difflow debug faulty.py --acceleration wegstein --on-error" \
   info "p reactor_feed" "p hot_feed" diagnose quit
$T 05-set-tear "$P" "stop-at iteration" continue "p tear" "set tear recycle.F_n_hexane 3.0" \
   "stop-at clear" continue continue
TERM_COLS=118 $T 06-json "$P --json" "break if residual<1e-6" continue "p recycle.F_n_hexane" \
   continue continue
python make_figures.py
