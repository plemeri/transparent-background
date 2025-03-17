import os
import shutil
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

cfg_path = os.environ.get(
    "TRANSPARENT_BACKGROUND_FILE_PATH", os.path.abspath(os.path.expanduser("~"))
)
os.makedirs(os.path.join(cfg_path, ".transparent-background"), exist_ok=True)

setuptools.setup(
    name="transparent-background",
    version="1.3.3",
    author="Taehun Kim",
    author_email="taehoon1018@postech.ac.kr",
    description="Make images with transparent background",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plemeri/transparent-background",
    packages=[
        "transparent_background",
        "transparent_background.modules",
        "transparent_background.backbones",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7.1",
        "torchvision>=0.8.2",
        "opencv-python>=4.6.0.66",
        "timm>=1.0.3",
        "tqdm>=4.64.1",
        "kornia>=0.5.4",
        "gdown>=4.5.4",
        "wget>=3.2",
        "easydict>=1.10",
        "pyyaml>=6.0",
        "albumentations>=1.3.1",
        # TODO: remove pin of albucore once this bug is fixed https://github.com/albumentations-team/albumentations/issues/1945
        "albucore>=0.0.16",
        "flet>=0.23.1",
        "pymatting>=1.1.13",
    ],
    extras_require={"webcam": ["pyvirtualcam>=0.6.0"]},
    entry_points={
        "console_scripts": [
            "transparent-background=transparent_background:console",
            "transparent-background-gui=transparent_background:gui",
        ],
    },
    include_package_data=True,
)
