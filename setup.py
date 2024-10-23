
from setuptools import setup, find_packages
setup(
  name = 'moyalstar',         # How you named your package folder (MyLib)
  packages = find_packages(),   # Chose the same as "name"
  version = '0.0.2',      # Start with a small number and increase it with every change you make
  license='GPL-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Moyal star-product SymPy calculator.``',   # Give a short description about your library
  author = 'Hendry Minfui Lim',                   # Type in your name
  author_email = 'hendryadi01@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/hendry24/moyalstar',   # Provide either the link to your github or to your website
  download_url = '',    # I explain this later on
  keywords = ['Moyal star-product'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'sympy>=1.13.3',
      ],
  python_requires='>=3',
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',   # Again, pick a license
  ],
)