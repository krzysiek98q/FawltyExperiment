#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import toml
from autoproject import FawltyInPractice

if __name__ == '__main__':
    packages_to_check = [
        'SQLAlchemy',
        'nltk', 
        'opencv-python', 
        'pandas',
        'plotly',
        'pygame',
        'scikit-learn',
        'scipy',
        'requests',
        'statsmodels'
        ]
    save_directory = '.repositories'
    save_result = 'results_ten_projects'
    os.mkdir(save_directory)
    os.mkdir(save_result)
    
    for package_name in packages_to_check:
        save_loc = f'{save_directory}/{package_name}'
        
        fawlty_exp = FawltyInPractice(package_name, save_loc)
        fawlty_exp.clone_package()
        
        out_check, err_check = fawlty_exp.run_fawlty_deps(['--json'])
        
        results = json.loads(out_check)
        del results['resolved_deps'], results['settings']
        
        keys_to_change = [
            'declared_deps',
            'imports',
            'undeclared_deps',
            'unused_deps'
            ]
        
        for key in keys_to_change:
            results[key] = [*map(lambda x: x['name'], results[key])]
        
        exp_info = {
            'project': {
                'name': package_name,
                'source_url': fawlty_exp.source_url,
                'version': results['version']
                },
            'experiment': {
                'imports': list(set(results['imports'])),
                'declared_deps': results['declared_deps'],
                'undeclared_deps': results['undeclared_deps'],
                'unused_deps': results['unused_deps']
                }
                
            }
        
        file_name = f'{save_result}/{package_name}.toml'
        with open(file_name, "w") as toml_file:
            toml.dump(exp_info, toml_file)
        