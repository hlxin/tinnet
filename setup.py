from setuptools import setup, find_packages

setup(
    name="tinnet",
    version="1.4.0",
    description="AI with Chemostructural Embeddings of Multimodality for Integrated Catalysis",
    author="Xin Group",
    author_email="hongliang06@gmail.com",
    url="https://github.com/hlxin/tinnet",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pymatgen",
        "shap",
        "ase",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "notebooks": ["jupyter", "nglview"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)