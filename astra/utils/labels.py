ztf_labels = {'CEP':['ACEP', 'DCEP', 'T2CEP'], 'RRLY':['RRab', 'RRc', 'RRd']}
ztf_subclass_map = {label:idx for idx, broader_class in enumerate(ztf_labels) for label in ztf_labels[broader_class]}
ztf_class_map = {broader_class:idx for idx, broader_class in enumerate(ztf_labels)}