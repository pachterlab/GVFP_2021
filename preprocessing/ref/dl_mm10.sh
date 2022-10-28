#!/bin/bash
# download the mouse reference

curl -O https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-mm10-2020-A.tar.gz
tar -xvf refdata-gex-mm10-2020-A.tar.gz
rm  refdata-gex-mm10-2020-A.tar.gz
