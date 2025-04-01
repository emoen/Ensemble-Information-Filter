#!/bin/bash

# Create the virtual environment if it doesn't exist
if [ ! -d "enif" ]; then
    python -m venv ensembleinfo 
    source ensembleinfo/bin/activate
    pip install -r requirements.txt
else
    source ensembleinfo/bin/activate
fi
