#!/bin/bash
echo "Installing dependencies"
pip install -q conda-build twine build anaconda-client

# Run the Python build module
rm -rf dist/*
version=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' src/sparkpolars/_version.py)
echo "Current version: $version"

IFS='.' read -r -a version_parts <<< "$version"

if [ "${version_parts[2]}" -lt 9 ]; then
    version_parts[2]=$((version_parts[2] + 1))
else
    version_parts[2]=0
    if [ "${version_parts[1]}" -lt 9 ]; then
        version_parts[1]=$((version_parts[1] + 1))
    else
        version_parts[1]=0
        version_parts[0]=$((version_parts[0] + 1))
    fi
fi

new_version="${version_parts[0]}.${version_parts[1]}.${version_parts[2]}"
echo "New version: $new_version"

# sed -i '' "s/^__version__ = \".*\"/__version__ = \"$new_version\"/" src/sparkpolars/_version.py

# echo "Building version: $new_version"
# python -m build

# echo "Uploading to PyPI"
# twine upload -p $TWINE_API_KEY dist/*

# echo "Building conda package"
# conda-build conda_receip/ --package-format=tar.bz2 --output-folder dist

# echo "Uploading to Anaconda"
# anaconda upload dist/noarch/sparkpolars-${new_version}-py_0.tar.bz2
