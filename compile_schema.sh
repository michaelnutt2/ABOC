#!/bin/bash
# Check if flatc is installed
if ! command -v flatc &> /dev/null
then
    echo "flatc could not be found. Please install FlatBuffers compiler."
    echo "On Linux: sudo apt install flatbuffers-compiler"
    echo "Or download from: https://github.com/google/flatbuffers/releases"
    exit 1
fi

# Compile the schema
echo "Compiling schema.fbs..."
flatc --python schema.fbs

echo "Done. Python classes generated in OctreeData/"
