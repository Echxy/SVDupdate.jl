# SVDupdate

[![Build Status](https://github.com/Echxy/SVDupdate.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/Echxy/SVDupdate.jl/actions/workflows/CI.yml?query=branch%3Amaster)
#   S V D u p d a t e . j l 
 

SVDupdate Algorithm based on  "Fast low-rank modifications of the thin singular
value decomposition" by Matthew Brand
in doi:10.1016/j.laa.2005.07.021

and based on "An answer to an open question in the incremental SVD" by Yangwen Zhang
in https://arxiv.org/pdf/2204.05398

The algorithm takes a already decomposed svd and updates to svd that results from adding 1 new row. 
This implementation uses eigen values/vector algorithm to save time and memory.
 
