#!/bin/bash

# Bulk import script for discovery productions
CASE_ID=$1
SOURCE_DIR=$2
DOC_TYPE=$3

if [ $# -ne 3 ]; then
    echo "Usage: $0 <case_id> <source_directory> <document_type>"
    echo "Document types: plaintiff_production, defendant_production, third_party_production, court_filings"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

TARGET_DIR="./discovery_sets/$CASE_ID/$DOC_TYPE"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Case directory $TARGET_DIR does not exist. Run setup_case.sh first."
    exit 1
fi

echo "Importing documents from $SOURCE_DIR to $TARGET_DIR"

# Copy files and maintain structure
rsync -av --progress "$SOURCE_DIR/" "$TARGET_DIR/"

echo "Import complete!"
echo "Files copied to: $TARGET_DIR"

# Optional: Trigger document processing
echo "To process these documents, run:"
echo "curl -X POST http://localhost:8000/api/cases/$CASE_ID/documents/bulk-process"