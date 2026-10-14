#!/usr/bin/env bash
# Reference files that are not graph input: every HPO annotation release (time-split comparator and the
# --gene-cut date filter) and HPO genes_to_disease.txt (causal-gene literature search in collect_v3.py,
# gene-symbol title topic in build_graph_v2.py --topic-genes). Existing files are kept; a download is
# moved into place only once complete.
#
#   bash pilot/scripts/fetch_reference_data.sh      # needs gh (authenticated) and curl
set -euo pipefail
cd "$(dirname "$0")/../.."
H=data/external_kg/hpoa_history
mkdir -p $H
fetch() { [ -s "$2" ] || { curl -sSfL -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
gh api "repos/obophenotype/human-phenotype-ontology/releases?per_page=100" --paginate \
  --jq '.[] | select(any(.assets[]; .name=="phenotype.hpoa")) | .tag_name' > $H/tags.txt
while read -r t; do
  fetch "https://github.com/obophenotype/human-phenotype-ontology/releases/download/$t/phenotype.hpoa" "$H/$t.hpoa"
done < $H/tags.txt
G=data/external_kg/genes_to_disease.txt
fetch https://purl.obolibrary.org/obo/hp/hpoa/genes_to_disease.txt $G
echo "hpoa releases $(ls $H/*.hpoa | wc -l), genes_to_disease $(wc -l < $G) lines"
