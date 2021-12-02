#from distutils.core import setup, Extension
from setuptools import setup, Extension
from Cython.Build import cythonize

module = Extension('CyWord2Vec',sources=['CyWord2Vec.pyx'])

setup(
    name='CyWord2Vec',
    version='1.0',
    author ='Pablo Rees',
    author_email='pablorees01@gmail.com',
    ext_modules=cythonize('CyWord2Vec.pyx', compiler_directives={'language_level': '3'})
)


