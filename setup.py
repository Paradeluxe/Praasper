from setuptools import setup, find_packages

setup(
    name='praasper',
    version='0.7.3',
    description='VAD-Enhanced ASR Framework for Researchers',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Tony Liu',
    author_email='paradeluxe3726@gmail.com',
    url='https://github.com/ParadeLuxe/Praasper',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchaudio',
        'jellyfish',
        'tqdm',
        'textgrid',
        'tiktoken',
        'funasr<1.3.3',
        'transformers',
        'scikit-learn',
        'librosa',
        'soundfile',
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10,<3.13',
)
