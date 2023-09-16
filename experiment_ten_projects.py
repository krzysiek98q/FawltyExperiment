#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import toml
from autoproject import FawltyInPractice

def report_undeclared(output, declared):
    """Creating the text of a markdown file that contains code fragments 
    that FawltyDeps recognized as undeclared dependencies
    
    Parameters
    ----------
    output
        FawltyDeps result.
        
    declared
        list of declared dependencies.
        
    """
    
    undeclared_report = ''
    base_results = json.loads(output)
    undeclared_report += '# undeclared dependenciec\n'
    
    for undec_dep in base_results['undeclared_deps']:
        name_dep =  undec_dep['name']
        references = undec_dep['references']
        dep_is_declared = name_dep in declared
        undeclared_report += f'## {name_dep}\n'
        
        if dep_is_declared:
            undeclared_report += f'<span style="color:red">**{name_dep} is in the dependencies declared**\n</span>.'
            
        for i, ref in enumerate(references):
            ref_path = ref['path']
            ref_line = ref['lineno']
            undeclared_report += f'### {i+1}.\n'
            undeclared_report += f'**path**: `{ref_path}`\n'
            undeclared_report += f'**line number**: {ref_line}\n'
            
            with open(ref_path, 'r') as pyfile:
                pycode = pyfile.readlines()
                pycode = pycode[ref_line - 2: ref_line + 5] if ref_line > 1 else pycode[ref_line - 1: ref_line + 5]
                pycode = ''.join(pycode)
                undeclared_report += f'```python\n{pycode}\n```\n'
                
    return undeclared_report


def report_warning(output):
    """Create the text of a markdown file that lists errors and warnings.
    
    Parameters
    ----------
    output
        FawltyDeps result errors and warnings.
        
    """
    
    warnring_report = ''
    warnring_report += '# warnings\n'
    if err_check.count('WARNING') > 0:
        warnings = output.split('\n')
        warnings = [*filter(lambda x: x.strip() != '', warnings)]
        for i, warning in enumerate(warnings):
            warnring_report += f'## {i+1}.\n'
            warnring_report += f'`{warning}`\n'
            
    return warnring_report 
            
if __name__ == '__main__':
    
    # list of libraries to test
    packages_to_check = [
        'tqdm',
        'nltk', 
        'pingouin', 
        'pandas',
        'seaborn',
        'Kivy',
        'scikit-learn',
        'scipy',
        'mlxtend',
        'statsmodels'
        ]
    
    mapping_names = {'scikit-learn': 'sklearn'}
    
    save_directory = '.repositories' # directory name for cloning repositories
    save_result = 'results_ten_projects' # name of the directory to save the results
    os.mkdir(save_directory)
    os.mkdir(save_result)
    
    for package_name in packages_to_check:
        os.mkdir(f'{save_result}/{package_name}')
        save_loc = f'{save_directory}/{package_name}'
        mapping = None if package_name not in mapping_names.keys() else mapping_names[package_name]
        
        # creation of an object for the automatic execution of
        # the experiment and cloning of the repository
        fawlty_exp = FawltyInPractice(package_name, save_loc, mapping_name=mapping)
        fawlty_exp.clone_package()
        
        # launching FawltyDeps
        out_check, err_check = fawlty_exp.run_fawlty_deps(['--json'])
        results = json.loads(out_check)
        
        
        # creation of a summary 
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
                f'{package_name}_version': fawlty_exp.package_pypi['info']['version']
                },
            
            'launching_fawltydeps': {
                'fawltydeps_version': results['version'],
                'settings': results['settings']
                },
            'experiment_result': {
                'number of warnings': err_check.count('WARNING'),
                'imports': list(set(results['imports'])),
                'declared_deps': results['declared_deps'],
                'undeclared_deps': results['undeclared_deps'],
                'unused_deps': results['unused_deps']
                }
                
            }
        
        # saving the summary result 
        file_name = f'{save_result}/{package_name}/summary.toml'
        with open(file_name, "w") as toml_file:
            toml.dump(exp_info, toml_file)
        
        # saving the file with found undeclared dependencies
        report = report_undeclared(out_check, results['declared_deps'])
        report_file_name = f'{save_result}/{package_name}/checking_of_undeclared.md'
        
        with open(report_file_name, "w") as md_file:
            md_file.write(report)
        
        # saving a file with a list of errors and warnings
        if err_check.count('WARNING') > 0:
            report = report_warning(err_check)
            report_file_name = f'{save_result}/{package_name}/warnings.md'
        
        with open(report_file_name, "w") as md_file:
            md_file.write(report)
        