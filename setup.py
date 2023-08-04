# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'tiktoken', 'tqdm',
    'wandb',
    "tokenizers>=0.13.3",
    "transformers>=4.29.2",
    "datasets>=2.12.0",
]

setuptools.setup(
    name="nanoGPT",
    version="1.0.1",
    author="Andrej Karpathy",
    description="The simplest, fastest repository for training/finetuning medium-sized GPTs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/nanoGPT",
    packages=setuptools.find_packages(include=['nanogpt_common']),
	license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ],
    include_package_data=True,
    install_requires=install_requires
)
