# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='LeafConductance',
      version='1.4',
      description='Detect changes in curve',
      author='Xavier Bouteiller',
      author_email='xavier.bouteiller@u-bordeaux.fr',
      packages=['LeafConductance'],
      install_requires=['kiwisolver>=1.3.1',
                        'loess>=2.0.11',
                        'matplotlib>=3.3.3',
                        'numpy>=1.19.4',
                        'pandas>=1.1.4',
                        'patsy>=0.5.1',
                        'Pillow>=8.0.1',
                        'plotbin>=3.1.3',
                        'pyparsing>=2.4.7',
                        'python-dateutil>=2.8.1',
                        'pytz>=2020.4',
                        'scikit-learn>=0.23.2',
                        'scipy>=1.5.4',
                        'six>=1.15.0',
                        'sklearn>=0.0',
                        'statsmodels>=0.12.1',
                        'threadpoolctl>=2.1.0'])



