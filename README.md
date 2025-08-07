# ASTRA: Attention-based Self-supervised Time-series Representation Architecture
## Working in DEV mode
```
git clone https://github.com/TorshaMajumder/astra.git
cd astra
python3 -m pip install -e .
```
## Creating Tensor Records
### When LSBD contain a "Class" column for labels
```
astra-data --target ../dataset/lyrae/ --path_to_buff ../dataset/lyrae/hats/zubercal_vrrlyr --min_detec 100 --train_size 0.80 --max_lcs_per_chunk 200
```
### When LSBD doesn't have a "Class" column or renaming the "Class" column
```
astra-data --target ../dataset/agn/ --path_to_buff ../dataset/agn/hats/zubercal_vagn --min_detec 100 --train_size 0.80 --max_lcs_per_chunk 200 --label "AGN"
```
## For more help!
```
astra-data --help
```
