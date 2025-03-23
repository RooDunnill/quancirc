from setuptools import setup, find_packages

setup(
    name="circuit",
    version="0.1",
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),  
    install_requires=["numpy", "scipy"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "quancirc_run=quancirc.src.scripts.quancirc_run:main",  # Calls "main()" inside "quancirc_run.py"
            "quancirc_checks=quancirc.src.scripts.quancirc_checks:main"
        ]}
)