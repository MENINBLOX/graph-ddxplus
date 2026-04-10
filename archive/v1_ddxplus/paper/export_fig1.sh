#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_FILE="${SCRIPT_DIR}/fig1_system_overview.html"
OUTPUT_FILE="${SCRIPT_DIR}/fig1.png"
TMP_FILE="/tmp/fig1_svg_only.html"

# Extract SVG from HTML and create a minimal page
python3 -c "
with open('${HTML_FILE}') as f:
    content = f.read()
start = content.find('<svg')
end = content.find('</svg>') + len('</svg>')
svg = content[start:end]
svg = svg.replace('width=\"960\" height=\"1200\"', 'width=\"1020\" height=\"1200\"')
html = '''<!DOCTYPE html>
<html><head>
<meta charset=\"UTF-8\">
<style>
  @import url(\"https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap\");
  * { margin: 0; padding: 0; }
  body { background: #fff; overflow: hidden; }
  svg { display: block; }
</style>
</head><body>
''' + svg + '''
</body></html>'''
with open('${TMP_FILE}', 'w') as f:
    f.write(html)
"

# Capture screenshot with headless Chrome
google-chrome \
  --headless \
  --disable-gpu \
  --force-device-scale-factor=1 \
  --screenshot="${OUTPUT_FILE}" \
  --window-size=1020,1300 \
  "file://${TMP_FILE}" \
  2>/dev/null

rm -f "${TMP_FILE}"
echo "Exported: ${OUTPUT_FILE}"
