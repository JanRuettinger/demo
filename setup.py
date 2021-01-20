import setuptools

setuptools.setup(
    name="example-pkg-jan_ruettinger", # Replace with your own username
    version="0.0.1",
    author="Jan Ruettinger",
    author_email="author@example.com",
    description="A package to load models and datasets for a demo",
    long_description="Doesn't exist.",
    long_description_content_type="text/markdown",
    url="https://github.com/JanRuettinger/demo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)