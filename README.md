# ASTRA: Attention-based Self-supervised Time-series Representation Architecture
## Working in DEV mode
```
>> git clone https://github.com/TorshaMajumder/astra.git
>> cd astra
>> python3 -m venv venv
>> source venv/bin/activate
## Optional upgrade
>> pip install --upgrade pip setuptools wheel pip-tools
## For CPU users
>> pip install -e .  
## For GPU users
>> pip install -e .[gpu]   

```
## Creating Tensor Records
### When LSBD contain a "Class" column for labels
```
astra-data --dest ../dataset/lyrae/ --path_to_buff ../dataset/lyrae/hats/zubercal_vrrlyr --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200
```
### When LSBD doesn't have a "Class" column or renaming the "Class" column
```
astra-data --dest ../dataset/agn/ --path_to_buff ../dataset/agn/hats/zubercal_vagn --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --label "AGN"
```
### If you want to delete some classes from LSDB
```
astra-data --dest ../dataset/cepheids/ --path_to_buff ../dataset/cepheids/hats/zubercal_vcep --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --del_label ACEP DCEP
```
### If you want to use/keep specific classes from LSDB
```
astra-data --dest ../dataset/cepheids/ --path_to_buff ../dataset/cepheids/hats/zubercal_vcep --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --keep_label ACEP DCEP T2CEP
```
## Training ASTRA framework with "Contrastive loss"
```
astra-net --loss contrastive --config ../config/contrastive-loss_triplet.yaml --epoch 100 --batch_size 300
```
## For more help!
```
astra-data --help
astra-net --help
```
