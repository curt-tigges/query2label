#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# security checks

echo "safety (failure is tolerated)"
FILE=requirements/prod.txt
if [ -f "$FILE" ]; then
    # We're in the main repo
    safety check -r requirements/prod.txt -r requirements/dev.txt
else
    # We're in the labs repo
    safety check -r ../requirements/prod.txt -r ../requirements/dev.txt
fi

echo "bandit"
bandit -ll -r {q2l_labeller} || FAILURE=true

# style checks

echo "pylint"
pylint q2l_labeller  || FAILURE=true

echo "pycodestyle"
pycodestyle q2l_labeller  || FAILURE=true

echo "pydocstyle"
pydocstyle q2l_labeller  || FAILURE=true

# type checking

echo "mypy"
mypy q2l_labeller  || FAILURE=true

# checks shell scripts for potential bugs

echo "shellcheck"
find . -name "*.sh" -print0 | xargs -0 shellcheck || FAILURE=true

# results

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
