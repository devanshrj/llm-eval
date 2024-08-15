import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="llm-eval",
    py_modules=["llm-eval"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(),
    # install_requires=[
    #     str(r)
    #     for r in pkg_resources.parse_requirements(
    #         open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    #     )
    # ],
    install_requires=['litellm', 'human-eval', 'humaneval-x'],
    dependency_links=[
        ''.join(['file:\\', os.path.join(os.getcwd(), 'human-eval#egg=human-eval-1.0')]),
        ''.join(['file:\\', os.path.join(os.getcwd(), 'humaneval-x#egg=humaneval-x-1.0')])
    ]
)
