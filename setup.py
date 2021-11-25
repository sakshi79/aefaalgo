import setuptools

with open("README.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'aefaalgo', 
  version = '0.0.5',     
  license='MIT',        
  description = 'Python package for AEFA: Artificial electric field algorithm for global optimization',
  long_description=long_description,
  long_description_content_type="text/markdown",  
  author = 'Sakshi Bhatia, Anupam Yadav',                  
  author_email = 'sakshisb@yahoo.com, anupuam@gmail.com',
  packages=setuptools.find_packages(),
  keywords = ['Optimization', 'Soft computing', 'Artificial intelligence', 'Electric force'],
  install_requires=[            
          'numpy>=1.19.5',
          'matplotlib>=2.2.2'
      ],
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=['wheel'],
    python_requires='>=3.6'
)