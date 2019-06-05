
from setuptools import setup

setup(name='dsbox-spen',
      version='1.0.0',
      url='https://github.com/rooshenas/dsbox-spen/tree/dsbox3',
      maintainer_email='Ke-Thia Yao',
      maintainer='kyao@isi.edu',
      description='DSBox-spen primitives',
      packages=[
          'dsbox',
          'dsbox.spen',
          'dsbox.spen.core',
          'dsbox.spen.primitives',
          'dsbox.spen.utils',
          ],
      python_requires='>=3.6',
      install_requires=[
          'scipy==1.2.1',
          'numpy>=1.11.1',
          'pandas>=0.20.1',
          'python-dateutil>=2.5.2',
          'six>=1.10.0',
          'stopit==1.1.2',
          'scikit-learn>=0.18.0',
          'tflearn',
          'Keras==2.2.4',
          'Pillow',
          'tensorflow-gpu',
          'h5py'
      ],
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'classification.multilabel_classifier.DSBOX = dsbox.spen.primitives:MLClassifier'
          ],
      }
)
