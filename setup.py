from setuptools import setup, find_packages

setup(
    name="tinnet",
    version="1.4.0",
    description="AI with Chemostructural Embeddings of Multimodality for Integrated Catalysis",
    author="Xin Group",
    author_email="hongliang06@gmail.com",  # Replace with your preferred contact
    url="https://github.com/hlxin/tinnet",  # Replace with the actual URL
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "torch>=1.10",  # Specify minimum version if needed
        "numpy",
        "pymatgen",
        "shap",  # Assuming SHAP explainability is used
        "ase",   # If Atomic Simulation Environment is required
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "notebooks": ["jupyter", "nglview"],
        "oc20": ["torch>=2.0", "lmdb", "torch-geometric"],
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
