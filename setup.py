from setuptools import setup, find_packages

setup(
    name='praasper',
version='0.8.1',
    description='VAD-Enhanced ASR Framework for Researchers',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Tony Liu',
    author_email='paradeluxe3726@gmail.com',
    url='https://github.com/ParadeLuxe/Praasper',
    packages=find_packages(),
    install_requires=[
        # NOTE: TestPyPI keeps stale copies of numpy/librosa/scipy/funasr/scikit-learn
        # that break install on Python 3.12. Full e2e install from TestPyPI fails
        # due to TestPyPI infrastructure — use `--no-deps` for structural verification
        # only. Real PyPI installs use the unconstrained bounds below.
        'numpy',
        'scipy',
        'torch',
        'torchaudio',
        'jellyfish',
        'tqdm',
        'textgrid',
        'tiktoken',
        'funasr>=1.3,<2.0',
        'transformers',
        'scikit-learn',
        'librosa',
        'soundfile',
        'static-ffmpeg>=3.0',
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10,<3.13',
)
