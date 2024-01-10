from setuptools import setup, find_packages

setup(
    name='minghao_utils',
    version='0.1',
    author='Minghao Fu',
    author_email='isminghaofu@gmail.com',
    description='Causality and Beyond',
    long_description=open('README.md').read(),
    url='https://github.com/yourusername/mypackage',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='example project',
    license='MIT',
    install_requires=[
        'somepackage>=1.0',
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    package_data={
        'mypackage': ['data/*.data'],
    },
    entry_points={
        'console_scripts': [
            'my-command=mypackage:main',
        ],
    },
)
