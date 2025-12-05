from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='inkling',
      version="0.0.12",
      description="Inkling BCI Models",
      license="MIT",
      author="InklingInc",
      author_email="hildieleyser@gmail.com",
      url="https://github.com/hildieleyser/Inkling",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
