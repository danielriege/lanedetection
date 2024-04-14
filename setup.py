import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='lanedetection',  
     version='0.0.1',
     author="Daniel Riege",
     description="Lanedetection pytorch model for tinycar suite.",
     license="MIT",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/danielriege/lanedetection",
     packages=["lanedetection", "lanedetection.models"],
     install_requires=[
        'torch>=2.0.0',
        'numpy>=1.22.0',
        'opencv-python>=4.5.5.62',
        'setuptools>=60.3.1',
     ],
     python_requires='>=3.8',
 )