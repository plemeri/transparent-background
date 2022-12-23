import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transparent-background",
    version="1.2.1",
    author="Taehun Kim",
    author_email="taehoon1018@postech.ac.kr",
    description="Make images with transparent background",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plemeri/transparent-background",
    packages=['transparent_background', 'transparent_background.modules', 'transparent_background.backbones'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['torch>=1.7.1', 'torchvision>=0.8.2', 'opencv-python>=4.6.0.66', 'timm>=0.6.11', 'tqdm>=4.64.1', 'kornia>=0.5.4', 'gdown>=4.5.4', 'pyvirtualcam>=0.6.0'],
    entry_points={
        "console_scripts": [
            "transparent-background=transparent_background:console",
        ],
    },
)
