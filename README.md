# RanPAC: Random Projections and Pre-trained Models for Continual Learning

## Official code repository for NeurIPS 2023 Accepted Paper:

Mark D. McDonnell, Dong Gong, Amin Parveneh, Ehsan Abbasnejad, Anton van den Hengel (2023). "RanPAC: Random Projections and Pre-trained Models for Continual Learning." Available at <https://arxiv.org/abs/2307.02251>.

Contact: <mark.mcdonnell@adelaide.edu.au>


### To reproduce results run code of the form

python main.py -i 7 -d cifar224

- for -i choose from  0,1,2,3,4,5,6,7

    - 0 is joint linear probe
    - 1 is joint full fine tuning
    - 2 is NCM, no PETL
    - 3 is RANPAC without RP and without PETL
    - 4 is RanPAC without PETL
    - 5 is NCM with PETL
    - 6 is RANPAC without RP
    - 7 is RanPAC

- for -d choose from 'cifar224', 'imageneta', 'imagenetr', 'cub', 'omnibenchmark', 'vtab', 'cars', 'core50', 'cddb', 'domainnet'
- except for cifar224, data will need to be downloaded and moved to relative paths at "../../Data/dataset_name/train/" and "../../Data/dataset_name/test/" -- see data.py

## Acknowlegment
This repo is based on aspects of https://github.com/zhoudw-zdw/RevisitingCIL

The implemenations of parameter-efficient tuning methods are based on [VPT](https://github.com/sagizty/VPT), [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), and [SSF](https://github.com/dongzelian/SSF).
