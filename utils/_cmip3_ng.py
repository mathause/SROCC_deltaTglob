from .common import CMIP_ng

# path of cmip5-ng data
folder_root = '/net/atmos/data/cmip3-ng/'

# available scenarios
scens = ['sresb1', 'sresa1b', 'sresa2']

cmip3_ng = CMIP_ng(folder_root=folder_root,
                   scens=scens,
                   cmip='cmip3')
