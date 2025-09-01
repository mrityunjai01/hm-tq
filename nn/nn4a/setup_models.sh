#!/bin/bash

# Script to create selected_model folder and copy model files sequentially

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/selected_model"

echo "Setting up selected_model directory..."

# Check and create folder if it doesn't exist
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating directory: $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
else
  echo "Directory already exists: $MODEL_DIR"
fi

# Define source model files
SOURCE_MODELS=("model1.py" "model2.py" "model3.py")

# Copy model files sequentially as model.py
for i in "${!SOURCE_MODELS[@]}"; do
  SOURCE_FILE="$SCRIPT_DIR/${SOURCE_MODELS[$i]}"
  TARGET_FILE="$MODEL_DIR/model.py"

  if [ -f "$SOURCE_FILE" ]; then
    echo "Copying ${SOURCE_MODELS[$i]} to model.py..."
    cp "$SOURCE_FILE" "$TARGET_FILE"
    echo "Copied: ${SOURCE_MODELS[$i]} -> $TARGET_FILE"
    python -m nn.nn4a.train

  else
    echo "Warning: Source file not found: $SOURCE_FILE"
  fi
done

ls -la "$MODEL_DIR"
