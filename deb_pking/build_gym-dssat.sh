#!/bin/bash

# TODO: grab gym-dssat version from sources
VERSION="0.0.8"
UNVERSIONED="gym-dssat-pdi"
DIRNAME="${UNVERSIONED}-${VERSION}"
TARBALL="${UNVERSIONED}_${VERSION}.orig.tar.gz"
DEBIAN="gym-dssat"

echo "PACKAGING ${DIRNAME} ($TARBALL)"

echo "cp -r ../${UNVERSIONED} ../${DIRNAME}"
cp -r ../${UNVERSIONED} ../${DIRNAME}

echo "tar zcf ../${TARBALL} ../${DIRNAME}"
tar zcf ../${TARBALL} ../${DIRNAME}

echo "cp -r ${DEBIAN} ../${DIRNAME}/debian"
cp -r ${DEBIAN} ../${DIRNAME}/debian

# TODO: document debian packages needed to build the package (dh...)
echo "cd ../${DIRNAME} && debuild --no-lintian -uc -us"
cd ../${DIRNAME} && debuild --no-lintian -uc -us

echo "cd .. && rm -rf ${DIRNAME}"
cd .. && rm -rf ${DIRNAME}
