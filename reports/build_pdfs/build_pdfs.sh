#!/usr/bin/env bash
# Regenerate the report PDFs in reports/ from their editable .md sources.
#
#   ./build_pdfs.sh                  rebuilds all three reports
#   ./build_pdfs.sh final_report     rebuilds just one
#
# Needs pandoc (brew install pandoc) and Google Chrome. The LMS upload
# copy (Group9_LogiEdge_Final.pdf, two levels up) is refreshed
# automatically whenever final_report is rebuilt.
set -euo pipefail
cd "$(dirname "$0")/.."   # work from reports/ so image paths in the .md resolve

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(final_report phase1_report phase2_report)

for name in "${TARGETS[@]}"; do
  pandoc "${name}.md" --standalone --css=build_pdfs/report_style.css \
    --metadata pagetitle="Group 9 — LogiEdge — ${name}" \
    -f markdown+smart -t html -o "_${name}_build.html"
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/${name}.pdf" "file://$PWD/_${name}_build.html" 2>/dev/null
  rm -f "_${name}_build.html"
  echo "built ${name}.pdf"
  if [ "$name" = "final_report" ]; then
    cp final_report.pdf ../../Group9_LogiEdge_Final.pdf
    echo "updated ../../Group9_LogiEdge_Final.pdf"
  fi
done
