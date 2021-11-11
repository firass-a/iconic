#!/bin/bash

VERSION="4.7"
SUBMOD="dssat-csm-data"
GITVER=$(echo $(git submodule status ../${SUBMOD}) | head -c 8 | tail -c 7)
UNVERSIONED="dssat-csm-data"
DIRNAME="${UNVERSIONED}-${VERSION}~${GITVER}"
TARBALL="${UNVERSIONED}_${VERSION}~${GITVER}.orig.tar.gz"
DEBIAN="dssat-csm-data"

echo "UPDATING GIT SUBMODULE"

echo "git submodule update ../${SUBMOD}"
git submodule update ../${SUBMOD}

echo "PACKAGING ${DIRNAME} ($TARBALL)"

echo "cp -r ../${SUBMOD} ../${DIRNAME}"
cp -r ../${SUBMOD} ../${DIRNAME}

echo "rm -rf ../${DIRNAME}/.git*"
rm -rf ../${DIRNAME}/.git*

echo "tar zcf ../${TARBALL} ../${DIRNAME}"
tar zcf ../${TARBALL} ../${DIRNAME}

echo "cp -r ${DEBIAN} ../${DIRNAME}/debian"
cp -r ${DEBIAN} ../${DIRNAME}/debian

# TODO: document debian packages needed to build the package (dh...)
echo "cd ../${DIRNAME} && debuild -uc -us"
cd ../${DIRNAME} && debuild -uc -us

echo "cd .. && rm -rf ${DIRNAME}"
cd .. && rm -rf ${DIRNAME}
