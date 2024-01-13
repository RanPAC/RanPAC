# RanPAC: Random Projections and Pre-trained Models for Continual Learning

## Official code repository for NeurIPS 2023 Published Paper:

Mark D. McDonnell, Dong Gong, Amin Parveneh, Ehsan Abbasnejad, Anton van den Hengel (2023). "RanPAC: Random Projections and Pre-trained Models for Continual Learning." Available at <https://arxiv.org/abs/2307.02251>.

Contact: <mark.mcdonnell@adelaide.edu.au>


### To reproduce results run code of the form

python main.py -i 7 -d cifar224

- for -i choose an integer between 0 and 16 inclusive

    - ViT-B/16 backbone:
        - 0 is joint linear probe (only implemented for CIL datasets, not DIL)
        - 1 is joint full fine tuning (only implemented for CIL datasets, not DIL)
        - 2 is NCM, no PETL
        - 3 is RANPAC without RP and without PETL
        - 4 is RanPAC without PETL
        - 5 is NCM with PETL
        - 6 is RANPAC without RP
        - 7 is RanPAC
    - ResNet50 Backbone (no PETL, i.e. no Phase 1):
        - 8 is NCM
        - 9 is RANPAC Phase 2 without RP
        - 10 is RanPAC Phase 2
    - ResNet152 Backbone (no PETL, i.e. no Phase 1):
        - 11 is NCM
        - 12 is RANPAC Phase 2 without RP
        - 13 is RanPAC Phase 2
    - CLIP ViT Backbone (no PETL, i.e. no Phase 1):
        - 14 is NCM
        - 15 is RANPAC Phase 2 without RP
        - 16 is RanPAC Phase 2

- for -d choose from 'cifar224', 'imageneta', 'imagenetr', 'cub', 'omnibenchmark', 'vtab', 'cars', 'core50', 'cddb', 'domainnet'
- except for cifar224, data will need to be downloaded and moved to relative paths at "./data/dataset_name/train/" and "./data/dataset_name/test/" -- see data.py

### Note on reproducibility

- For different seeds and class sequences, there can be in the order of +/-1% differences in final average accuracy, and this can affect which PETL method performs best. We did not have time or the motivation to run many repeats of each PETL method and statistically choose the best. What matters is that PETL methods generally boost performance and the choice of PETL method is dataset-dependent.
- Based on the above, our args files only contain arguments for one choice of PETL method for each dataset. These do not necessarily reflect the choice used in the original version of the paper, but the final version does use the arguments contained in this repository.

## Datasets tested on in McDonnell et al.

Five of the datasets tested on are specific splits and/or subsets of the full original datasets. These versions were created by Zhou et al in:

    @article{zhou2023revisiting,
        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
        journal = {arXiv preprint arXiv:2303.07338},
        year = {2023}
    }

- The following links are copied verbatim from the README.md file in the github repository of Zhou et al at https://github.com/zhoudw-zdw/RevisitingCIL:

> **CUB200**:  Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)  
> **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)  
> **ImageNet-A**:Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)  
> **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)  
> **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)  

- All remaining datasets use standard train and test splits as described in McDonnell et al.


## Acknowlegment
This repo is based on aspects of https://github.com/zhoudw-zdw/RevisitingCIL

The implemenations of parameter-efficient tuning methods are based on [VPT](https://github.com/sagizty/VPT), [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), and [SSF](https://github.com/dongzelian/SSF).
