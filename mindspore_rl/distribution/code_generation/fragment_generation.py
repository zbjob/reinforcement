"""
Generate fragment
"""

import sys
import os
import shutil
import importlib
from .annotation_parser import interface_parser
from .generate_code import generate_fragment


def fragment_generation(algorithm, algorithm_config, policy_name):
    '''Generate fragments'''
    path = os.path.dirname(os.path.abspath(policy_name))
    src_template = path+'/template.txt'
    template = path+'/template.py'
    shutil.copy(src_template, template)
    sys.path.insert(0, path)
    policy_module = importlib.import_module(policy_name)
    policy = getattr(policy_module, policy_name)()
    if policy.auto:
        parameter_list = interface_parser(policy)
        position = {'Trainer': 'train_one_episode'}
    else:
        algorithm, parameter_list = annotation_parser.anno_parser(algorithm)
    fragments = generate_fragment(algorithm, parameter_list, template, algorithm_config, position, policy)
    print(fragments)
    return fragments
