#!/bin/bash
# Clean up failed 10-sample test artifacts

set -e

echo "Cleaning up failed 10-sample experiment artifacts..."
echo ""

OLD_DIR="artifacts/cogvideox15_10samples_test"

if [ -d "$OLD_DIR" ]; then
    echo "Found old test directory: $OLD_DIR"
    ls -lh "$OLD_DIR"
    echo ""

    read -p "Delete this directory? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OLD_DIR"
        echo "✓ Deleted $OLD_DIR"
    else
        echo "Skipped deletion"
    fi
else
    echo "No old test directory found"
fi

echo ""
echo "Done!"
