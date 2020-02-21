#!/bin/bash

# .travis_script.sh - a script to allow conditional installs of
# acnportal dependencies.

# Set USE_GYM environment variable to default of True if it is not set.
: "${USE_GYM:=true}"

if $USE_GYM; then
  pip install .[gym]
else
  pip install .
fi
