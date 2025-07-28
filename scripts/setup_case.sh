#!/bin/bash

# Setup script for new case initialization
CASE_ID=$1

if [ -z "$CASE_ID" ]; then
    echo "Usage: $0 <case_id>"
    exit 1
fi

echo "Setting up new case: $CASE_ID"

# Create case directory structure
CASE_DIR="./discovery_sets/$CASE_ID"
mkdir -p "$CASE_DIR"/{plaintiff_production,defendant_production,third_party_production,court_filings}

# Create metadata directory
mkdir -p "./data/case_metadata/$CASE_ID"

# Create analysis output directories
mkdir -p "./analysis_outputs/contradictions/$CASE_ID"
mkdir -p "./analysis_outputs/timelines/$CASE_ID"
mkdir -p "./analysis_outputs/privilege_logs/$CASE_ID"
mkdir -p "./analysis_outputs/evidence_summaries/$CASE_ID"

echo "Case $CASE_ID setup complete!"
echo "Directories created:"
echo "  - $CASE_DIR"
echo "  - ./data/case_metadata/$CASE_ID"
echo "  - Analysis output directories"