#!/bin/bash

# Script para descargar el dataset de Heart Disease

echo "════════════════════════════════════════════════════════"
echo "  Downloading Heart Disease Dataset"
echo "════════════════════════════════════════════════════════"

# URL del dataset procesado (Cleveland database)
DATASET_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_FILE="heart.csv"

if ! command -v wget &> /dev/null; then
    echo "ERROR: wget is not installed"
    echo "Please install wget or download manually from:"
    echo "  $DATASET_URL"
    exit 1
fi

echo "Downloading dataset..."
wget -O "$OUTPUT_FILE" "$DATASET_URL"

if [ $? -eq 0 ]; then
    echo "✓ Dataset downloaded successfully: $OUTPUT_FILE"

    # Agregar header al CSV
    echo "Adding header..."
    sed -i '1i\age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num' "$OUTPUT_FILE"

    echo "✓ Dataset ready!"
    echo ""
    echo "Dataset info:"
    echo "  - Samples: ~303"
    echo "  - Features: 13"
    echo "  - Target: num (0-4, will be converted to binary)"
    echo ""
    echo "Next steps:"
    echo "  1. mkdir build && cd build"
    echo "  2. cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  3. make"
    echo "  4. ./heart_disease_train ../heart.csv"
else
    echo "✗ Download failed"
    echo ""
    echo "Manual download instructions:"
    echo "  1. Visit: https://archive.ics.uci.edu/ml/datasets/heart+disease"
    echo "  2. Download 'processed.cleveland.data'"
    echo "  3. Rename to 'heart.csv'"
    echo "  4. Add header line:"
    echo "     age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num"
    exit 1
fi