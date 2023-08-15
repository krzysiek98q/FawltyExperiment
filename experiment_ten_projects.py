#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

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
        
        out_check, err_check = fawlty_exp.run_fawlty_deps()
        out_imports, err_imports = fawlty_exp.run_fawlty_deps(fawlty_optins=['--list-imports'])
        
        out_check = out_check.replace('For a more verbose report re-run with the `--detailed` option.', '')
        out_imports = out_imports.replace('For a more verbose report re-run with the `--detailed` option.', '')
        
        report = out_check
        report += f'All imports:\n{out_imports}'
        
        with open(f'{save_result}/{package_name}.txt', 'w') as file:
            file.write(report)
        