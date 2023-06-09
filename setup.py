from setuptools import setup

setup(
    name='ml_draftpick_dss',
    version='0.1.0',    
    description='ML Draftpick DSS',
    url='https://github.com/R-N/ml_draftpick_dss',
    author='Muhammad Rizqi Nur',
    author_email='rizqinur2010@gmail.com',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy', 
        'tensorflow',
        'tensorflow-addons',
        'paddleocr',
        'paddlepaddle',
        'thefuzz',
        'requests-html',
        'torch',
        'opencv-python',
        'scikit-image',
        'albumentations>=1.3.0',
        'plotly',
        'torchinfo',
        'optuna',
        'scipy',
        'torch-lr-finder',
        'catboost',
        'coloredlogs',
        'tqdm'
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)