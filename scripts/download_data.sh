#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Downloads a list of files into the project's `data/` directory.
# Safe, retrying curl with redirects enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
mkdir -p "$DATA_DIR"

urls=(
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=MCE_Bioactive_Compounds_HEK293T_10%CE%BCM_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=MCE_Bioactive_Compounds_HEK293T_10%CE%BCM_MetaData.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=MCE_Bioactive_Compounds_MDA_MB_231_10%CE%BCM_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=MCE_Bioactive_Compounds_MDA_MB_231_10%CE%BCM_MetaData.xlsx"
	"https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-025-02781-5/MediaObjects/41592_2025_2781_MOESM3_ESM.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_HEK293T_10_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_HEK293T_10_MetaData.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_HEK293T_20_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_HEK293T_20_MetaData.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_MDA_MB_231_10_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_MDA_MB_231_10_MetaData.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_MDA_MB_231_20_Counts.xlsx"
	"https://cigs.iomicscloud.com/cigs/cigs_doc.php?url=TCM_Compounds_MDA_MB_231_20_MetaData.xlsx"
)

download_file() {
	local url="$1"
	# extract the filename from the query param `url=` if present, otherwise fallback to basename
	local fname
	if [[ "$url" =~ url=([^&]+) ]]; then
		fname="${BASH_REMATCH[1]}"
	else
		fname="$(basename "$url")"
	fi
	local out="$DATA_DIR/$fname"

	if [ -f "$out" ]; then
		echo "Skipping existing file: $out"
		return 0
	fi

	echo "Downloading $fname to $out"
	curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 10 -o "$out" "$url"
}

for u in "${urls[@]}"; do
	download_file "$u"
done

echo "All downloads complete. Files saved in: $DATA_DIR"
