import ConfigParser

config = ConfigParser.RawConfigParser()

# When adding sections or items, add them in the reverse order of
# how you want them to be displayed in the actual file.
# In addition, please note that using RawConfigParser's and the raw
# mode of ConfigParser's respective set functions, you can assign
# non-string values to keys internally, but will receive an error
# when attempting to write to a file or when you get it in non-raw
# mode. SafeConfigParser does not allow such assignments to take place.
config.add_section('data')
config.set('data', 'dataset', 'AWA')
config.set('data', 'nfold', '5')
config.set('data', 'AWA_DIR', '../data/AWA/')
config.set('data', 'norm_method', 'L2')


config.add_section('model')
config.set('model', 'method', 'sync_memory')
config.set('model', 'lamda', '0.0005')
config.set('model', 'sim_scale', '1')

# Writing our configuration file to 'example.cfg'
with open('zero-shot.cfg', 'wb') as configfile:
    config.write(configfile)
