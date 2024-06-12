from setuptools import setup, find_packages

setup(
    name='minimal_frame',
    version='0.0.1',
    packages=find_packages(include=['minimal_frame', 'minimal_frame.*']),
    install_requires=[
        'numpy',
        'numba',
        'sympy'
    ],
    include_package_data=True,
    description='Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency',
    long_description=open('README.md').read(),
    url='https://github.com/divelab/MFA',
    author='Yuchao Lin',
    author_email='kruskallin@tamu.edu',
    license='MIT',
    python_requires='>=3.9',
)