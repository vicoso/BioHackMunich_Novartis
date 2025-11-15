#!/bin/bash

# Script to loop through DRIVE_TARGETS and download each file individually using uv run

# Define the targets (extracted from download_preprocessed_data.sh)
DRIVE_TARGETS="https://drive.google.com/file/d/188NMF8IlvoqgmTGSRH--cWvlWvUAILDm/view?usp=sharing,https://drive.google.com/file/d/1MtKQLNwT4f887_skr9A7HhKRIcNUUiuB/view?usp=sharing,https://drive.google.com/file/d/1nQNYOdyGHjmKjkO9ma3CRK21D5lWTR_K/view?usp=sharing,https://drive.google.com/file/d/1Sxox92nOpxLVA3RqkCXH5oLl8al8frhb/view?usp=sharing,https://drive.google.com/file/d/10BMAiSrID9YXz1fzZrBWeBAHVdJibcOg/view?usp=sharing,https://drive.google.com/file/d/1Equ3Z4pg-29IjLLQWmP5RJuJ01_Tj358/view?usp=sharing,https://drive.google.com/file/d/15V6IQXNDm4T88UacnGkmJqD2BECoPWAI/view?usp=sharing,https://drive.google.com/file/d/1lsR_VequOMOInO-Ns1XOzbIYwdVh-MHT/view?usp=sharing,https://drive.google.com/file/d/1RGlAhDERMAOiJRxB7-P4dWQoTepyTH8K/view?usp=sharing,https://drive.google.com/file/d/1xpwTFhkVp2lTxnnC_5VwxLMdTfZjPZQb/view?usp=sharing,https://drive.google.com/file/d/1zJiWMO4O7EBZFg6WrzkPNAI95C26V3nW/view?usp=sharing,https://drive.google.com/file/d/1YZ6VH2JJU37_d_ExOxgzFkSNEluawhwf/view?usp=sharing,https://drive.google.com/file/d/1UbIP4n0hnmPDN2dqn1HcurQxv5tIl-uw/view?usp=sharing,https://drive.google.com/file/d/11yiIM-oy0VTw2t1EYW0stzHBeNu4m-7I/view?usp=sharing"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

echo "Starting download of all files using uv run..."
echo "Project directory: $PROJECT_DIR"
echo "Output directory: $PROJECT_DIR/data"
echo

# Convert comma-separated targets into array
IFS=',' read -ra TARGETS <<< "$DRIVE_TARGETS"

# Counter for progress tracking
counter=0
total=${#TARGETS[@]}

# Loop through each target
for target in "${TARGETS[@]}"; do
    counter=$((counter + 1))
    
    echo "[$counter/$total] Downloading file from: $target"
    
    # Run the Python download script using uv
    if uv run scripts/gdrive_public.py --target "$target"; then
        echo "✓ Successfully downloaded file $counter"
    else
        echo "✗ Failed to download file $counter: $target"
        echo "Exit code: $?"
    fi
    
    echo "----------------------------------------"
    
    # Optional: Add a small delay between downloads to be nice to Google's servers
    sleep 2
done

echo
echo "Download process completed!"
echo "Files saved to: $PROJECT_DIR/data/"

# List downloaded files
if [ -d "$PROJECT_DIR/data" ]; then
    echo
    echo "Downloaded files:"
    ls -la "$PROJECT_DIR/data/"
else
    echo "Note: data directory not found - downloads may have failed"
fi