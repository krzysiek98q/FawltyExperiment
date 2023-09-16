#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re

from subprocess import Popen, PIPE
import subprocess

import requests
import toml
from git import Repo

class FawltyInPractice:
    """Uses PyPI API to automatically check package dependencies using FawltyDeps.
    
    Parameters
    ----------
    package_name: string
        The name of the package from The Python Package Index (PyPI) repository. 
        It should be the full name of the package, not a mapped name, 
        for example 'beautifulsoup4', not 'bs4'.
        
    save_location: string
        The name of the working directory into which the package repository will be cloned.
    
    save_requires: bool
        Information whether to create a 'requirements.txt' file based on the PyPI repository API
        when cloning the repository.
        
    no_repo_address: {'error', 'manual'}
        How to deal with missing repository address in API PyPI:
            - 'error' - do not execute codes and return an exception;
            - 'manual' - will ask you to manually enter the address of the repository.
        
    """
    
    def __init__(
            self, 
            package_name: str,
            save_location: str,
            save_requires: bool=True,
            no_repo_address: str='error'
            ):
        
        self.package_name = package_name
        self.save_location = save_location
        self.save_requires = save_requires
        
        assert no_repo_address in ['error', 'manual']
        self.no_repo_address = no_repo_address
        
        self.__get_pypi_info()
        
        self.venv_loc = f'{self.save_location}/venv'
        self.venv_python = f'{self.venv_loc}/bin/python3'
    
    
    def clone_package(
            self, 
            create_venv:bool=True, 
            install_requires:bool=True,
            install_fawlty:bool=True
            ):
        """Clones the library repository. The url of the package repository 
        is searched for in PyPI. This is the shortest string of url found, 
        which includes github.com or gitlab.com.
        
        Parameters
        ----------
        create_venv: bool
            Decision whether to install a virtual environment after 
            cloning the package. For automatic package testing to make sense,
            this parameter should be set to True.
            
        install_requires: bool
            Parameter to determine whether to install the requirements found in PyPI.
        
        install_fawlty: bool
            Decision whether, after creating a virtual environment, to install FawltyDeps in it.
        """
        
        Repo.clone_from(self.source_url, self.save_location)
        
        if self.save_requires:
            requires_text = '\n'.join(self.pypi_requires)
            requires_path = f'{self.save_location}/requirements.txt'
            with open(requires_path, 'w') as file:
                file.write(requires_text)
        if create_venv:
            create_cmd = ["python3", '-m', "venv", self.venv_loc]
            self.venv_out, self.venv_err = self.__terminal(create_cmd)
            
            if install_fawlty:
                install_fd_cmd = [self.venv_python, '-m', 'pip', 'install', 'fawltydeps']
                self.fawlty_install_out, self.fawlty_install_err = self.__terminal(install_fd_cmd)
                
            if install_requires:
                install_req_cmd = [self.venv_python, '-m', 'pip', 'install', 'requires_text']
                self.req_install_out, self.req_install_err = self.__terminal(install_req_cmd)
    
    
    def run_fawlty_deps(self, fawlty_options:list=None):
        """Starts FawltyDeps at the location of the package's working directory. 
        
        Parameters
        ----------
        fawlty_options: bool
            Optional options for running fawlty deps. In the absence of additional options, 
            runs the command: 'python_loc -m fawltydeps work_dir'.
        
        """
        
        fawlty_cmd = [self.venv_python, '-m', 'fawltydeps', self.save_location]
        if fawlty_options is not None:
            fawlty_cmd.extend(fawlty_options)
            
        return self.__terminal(fawlty_cmd)
        
        
    def __get_pypi_info(self):
        """Retrieves information from the PyPI API and extracts the most 
        important from the point of view of the experiment
        """
        
        url = f'https://pypi.org/pypi/{self.package_name}/json'
        req = requests.get(url).text
        self.package_pypi = json.loads(req)
        
        package_urls = self.package_pypi['info']['project_urls'].values()
        package_urls = [*filter(lambda x: bool(re.search(r'github\.com|gitlab\.com', x)), package_urls)]
        
        if len(package_urls) > 0:
            self.source_url = min(package_urls, key=len) 
            self.source_url = self.source_url[:-1] if self.source_url[-1] == '/' else self.source_url
            self.source_url += '.git'
            
        else:
            if self.no_repo_address == 'error':
                raise Exception('no address in the PyPI repository')
                
            elif self.no_repo_address == 'manual':
                self.source_url = input(
                    'Specify the address of the library\'s repository (with .git at the end): '
                    )
            
        
        pypi_requires = self.package_pypi['info']['requires_dist']
        self.pypi_requires = [*map(lambda x: re.sub(r';\s.*', '', x), pypi_requires)] if pypi_requires is not None else []
        
        
    def __terminal(self, command:list):
        """Runs passed commands in bash and returns output and a runtime error."""
        
        command = [*map(lambda x: x.strip(), command)]
        exec_command = Popen(
            command,
            stdin=PIPE,
            stdout=PIPE,  
            stderr=PIPE)
        output, err = exec_command.communicate()
        output, err = output.decode('utf-8'), err.decode('utf-8')
        return output, err 
        

    
