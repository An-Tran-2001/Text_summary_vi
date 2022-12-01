import os
import sys
from setuptools import setup, find_packages

path = os.getcwd()
sys.path.append(path)
setup(
    name='Text_summary',
    version='0.1',
    description='Text summarization',
    author = 'Siddharth',
    author_email = '    ',
    url = '   ',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'nltk',
        'scikit-learn',
        'gensim',
        'networkx',
        'matplotlib',
        'nltk',
        'pandas',
        'seaborn',
        'spacy',
        'pyLDAvis',
        'pytextrank',
        'sumy',
        'summa',
    ],
)

# Path: Text_summary\text_summary\__init__.py