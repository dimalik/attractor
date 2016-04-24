import os
import ConfigParser

# config = ConfigParser.RawConfigParser()

# config.add_section('CONSTANTS')
# config.set('CONSTANTS', 'input_dims', '30')
# config.set('CONSTANTS', 'data_path', 'data')
# config.set('CONSTANTS', 'binomial_probability', '0.3')
# config.set('CONSTANTS', 'max_iterations', '100')

# config.add_section('ATTRACTOR')
# config.set('ATTRACTOR', 'nb_ticks', '20')
# config.set('ATTRACTOR', 'tau', '0.2')
# config.set('ATTRACTOR', 'train_for', '10')
# config.set('ATTRACTOR', 'epochs', '50')
# config.set('ATTRACTOR', 'init_min', '0')
# config.set('ATTRACTOR', 'init_max', '0.1')

# with file('config', 'wb') as configfile:
#     config.write(configfile)

config = ConfigParser.ConfigParser({
    'input_dims': '30',
    'data_path': 'data',
    'binomial_probability': '0.3',
    'max_iterations': '100',
    'nb_ticks': '20',
    'tau': '0.2',
    'train_for': '10',
    'epochs': '50',
    'init_min': '0.0',
    'init_max': '0.1'})

config.read('config')

options = {
    'input_dims': config.getint('CONSTANTS', 'input_dims'),
    'data_path': os.path.realpath(config.get('CONSTANTS', 'data_path')),
    'binomial_probability': config.getfloat('CONSTANTS',
                                            'binomial_probability'),
    'max_iterations': config.getint('CONSTANTS', 'max_iterations'),
    'nb_ticks': config.getint('ATTRACTOR', 'nb_ticks'),
    'tau': config.getfloat('ATTRACTOR', 'tau'),
    'train_for': config.getint('ATTRACTOR', 'train_for'),
    'epochs': config.getint('ATTRACTOR', 'epochs'),
    'init_min': config.getfloat('ATTRACTOR', 'init_min'),
    'init_max': config.getfloat('ATTRACTOR', 'init_max')
}
