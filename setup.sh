#!/bin/bash

# chmod +x setup.sh

mkdir sign_language_translator
cp Week\ 5/* sign_language_translator
cp Week\ 6/* sign_language_translator
cp -r data/asl_citizen_processed sign_language_translator
mkdir -p sign_language_translator/saved_models
mkdir -p sign_language_translator/saved_plots