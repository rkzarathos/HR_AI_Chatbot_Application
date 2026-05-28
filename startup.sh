#!/bin/bash
set -e

echo "Building Chroma index..."
python scripts/build_chroma_index.py

echo "Building PageIndex..."
python scripts/build_pageindex.py
