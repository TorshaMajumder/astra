#
# Import all dependencies
#
import os
from setuptools import setup, find_packages


def get_version(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                quotes_type = '"' if '"' in line else "'"
                version = line.split(quotes_type)[1]
                return version
    raise RuntimeError("Unable to find version string.")

def get_lines(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


def get_valid_packages_from_requirement_file(file_path):
    lines = get_lines(file_path)
    # Filter out non-package lines that are legal for `pip install -r` but fail for setuptools' `require`:
    pkg_list = [p for p in lines if p.lstrip()[0].isalnum()]
    return pkg_list



setup(
    name='dart',
    version= get_version(os.path.join('dart', '__init__.py')),
    packages=find_packages(),  # Automatically finds your package
    install_requires=get_valid_packages_from_requirement_file("requirements.txt"),
    author='Your Name',
        author_email='your_email@example.com',
        description='A short description of your package',
        long_description='A more detailed description (optional)',
        url='https://your-package-url.com',
    entry_points={
        'console_scripts': [
            'dart-data = scripts.generate_data:main', # Entry point for the command line
            'dart-transformer = scripts.training:main',

        ],
    },
)

