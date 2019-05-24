from .common import CMIP_ng

# path of cmip5-ng data
folder_root = '/net/atmos/data/cmip5-ng/'

# available scenarios
scens = ['rcp26', 'rcp45', 'rcp60', 'rcp85']

cmip5_ng = CMIP_ng(folder_root=folder_root,
                   scens=scens,
                   cmip='cmip5')
