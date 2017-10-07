# from distutils.core import setup
from setuptools import setup

import os
from setuptools import setup  # , Command

#
# class CleanCommand(Command):
#     """Custom clean command to tidy up the project root."""
#     user_options = []
#
#     def initialize_options(self):
#         pass
#
#     def finalize_options(self):
#         pass
#
#     def run(self):
#         os.system('rm -vrf ./build ./*.pyc ./*.tgz ./*.egg-info')
#         os.system('tar -xvzf ./dist/ect-0.1.tar.gz  -C ./dist')
#         os.system('rm -vrf ./dist/ect-0.1.tar.gz')
#

setup(name="ect",
      version="0.1",
      packages=['eclustering',
                'eclustering.agglomerative',
                'eclustering.divisive',
                'eclustering.ik_means',
                'eclustering.pattern_initialization',
                'generators'],
      author="Petr Eremeykin",
      author_email="eremeykin@gmail.com",
      install_requires=['matplotlib',
                        'pyqt5',
                        'scikit-learn',
                        'scipy',
                        'numpy',
                        'pandas']
                        'pandas'],
      # cmdclass={
      #     'clean': CleanCommand,
      # }
      )



# tests_require = [
#     'mock',
#     'nose',
#     ]
