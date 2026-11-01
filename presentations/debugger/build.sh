#!/usr/bin/env bash
# Build the deck: an HTML version that plays the recordings, and a PDF that shows
# each recording's last frame instead (a PDF cannot animate). Needs node; the PDF
# needs a Chrome or Chromium (CHROME_PATH).
set -euo pipefail
cd "$(dirname "$0")"
MARP="npx --yes @marp-team/marp-cli@4.1.2 --no-stdin"
$MARP debugger.md --html --allow-local-files -o debugger.html
sed -e 's/\(assets\/[0-9a-z-]*\)\.gif)/\1.png)/g' \
    -e 's#<video src="\([^"]*\)\.mp4" poster="\([^"]*\)"[^>]*></video>#<img class="shot" src="\2">#' \
    debugger.md > _static.md
$MARP _static.md --html --allow-local-files --pdf -o debugger.pdf
if [ "${PNG:-}" = 1 ]; then $MARP _static.md --html --allow-local-files --images png -o _preview/s.png; fi
rm -f _static.md
