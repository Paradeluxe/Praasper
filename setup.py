from setuptools import setup, find_packages

setup(
    name='praasper',      # 在PyPI上唯一的包名
    version='0.4.0',              # 版本号，遵循语义化版本规范
    description='VAD-Enhanced ASR Framework for Researchers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tony Liu',
    author_email='paradeluxe3726@gmail.com',
    url='https://github.com/ParadeLuxe/Praasper',
    packages=find_packages(),     # 自动查找包
    install_requires=[            # 你的项目所依赖的第三方库
        'textgrid',
        'librosa',
        'praat-parselmouth',
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',      # 指定Python版本要求
)