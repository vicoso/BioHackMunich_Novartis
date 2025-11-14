#!/bin/bash
# Script: download_excel_files.sh
# Description: Fetches the HTML content from the target URL, extracts all links
# that end in .xlsx or .xls, constructs the full URL for each file, and
# downloads them into the current directory.

# --- Configuration ---
# The base URL of the website we are scraping
TARGET_URL="https://cigs.iomicscloud.com/"

# The base domain (used for constructing full URLs from relative paths)
BASE_DOMAIN="https://cigs.iomicscloud.com"

# Temporary file to store the list of links
TEMP_LINKS_FILE="/tmp/excel_links_$$"

# --- Main Logic ---

echo "--- Starting Excel File Scraper ---"
echo "Targeting URL: ${TARGET_URL}"

# 1. Fetch the webpage content and extract potential Excel links
# We use a combination of tools:
# - curl: Fetches the HTML content.
# - grep: Filters for lines containing 'href=' and '.xls' or '.xlsx' (case insensitive).
# - sed: Uses a regular expression to extract just the content within the quotes of the href attribute.
echo "1. Fetching HTML and extracting raw link paths..."
curl -s "${TARGET_URL}" | \
    grep -Eoi 'href="[^"]+\.(xlsx|xls)"' | \
    sed 's/.*href="\([^"]*\)".*/\1/' > "${TEMP_LINKS_FILE}"

# Check if any links were found
if [ ! -s "${TEMP_LINKS_FILE}" ]; then
    echo "No Excel files (.xlsx or .xls) were found in the raw HTML links."
    rm -f "${TEMP_LINKS_FILE}"
    exit 1
fi

echo "Found $(wc -l < "${TEMP_LINKS_FILE}") potential Excel links."
echo "2. Processing and downloading files..."

# 2. Loop through the found links, construct the full URL, and download
while IFS= read -r RELATIVE_PATH; do
    # Determine the full URL
    if [[ "${RELATIVE_PATH}" == http* ]]; then
        # Link is already absolute
        FULL_URL="${RELATIVE_PATH}"
    elif [[ "${RELATIVE_PATH}" == /* ]]; then
        # Link is domain-relative (starts with /), prepend the base domain
        FULL_URL="${BASE_DOMAIN}${RELATIVE_PATH}"
    else
        # Link is path-relative (does not start with /), prepend the full target URL
        # NOTE: This assumes the target URL ends in a slash for correct concatenation.
        FULL_URL="${TARGET_URL}${RELATIVE_PATH}"
    fi

    # Extract the filename for use with curl -O
    FILENAME=$(basename "${FULL_URL}")

    echo "Downloading: ${FILENAME} from ${FULL_URL}"

    # Use curl to download the file, using the original filename (-O)
    curl -s -O "${FULL_URL}"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded ${FILENAME}."
    else
        echo "ERROR: Failed to download ${FILENAME}."
    fi

done < "${TEMP_LINKS_FILE}"

# 3. Cleanup
rm -f "${TEMP_LINKS_FILE}"
echo "--- Script Finished ---"

exit 0