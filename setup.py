from setuptools import find_packages, setup


def get_requirements(file_path: str):
    """Read requirements.txt and return list of dependencies, excluding -e ."""
    requirements = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and line != "-e .":
                requirements.append(line)
    return requirements


setup(
    name="fake-news-detection",
    version="0.1.0",
    author="Harsh",
    author_email="nimsatkarharsh@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
