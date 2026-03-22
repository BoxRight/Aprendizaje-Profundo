#!/bin/sh
# Reassemble Dataset_848.7z from split parts.
set -e
cd "$(dirname "$0")"
cat $(ls Dataset_848.7z.part* | LC_ALL=C sort) > Dataset_848.7z
echo "OK: wrote Dataset_848.7z — run: 7z t Dataset_848.7z"
