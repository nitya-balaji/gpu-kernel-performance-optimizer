from setuptools import find_packages, setup

#creating this function to return the list of requirements
def get_requirements(file_path:str) -> list[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines() #reading the object
        requirements=[req.replace("\n", "") for req in requirements] #removing the "\n" to replace with a blank for each element within requirements, then putting each element into a list
        if "-e ." in requirements:
           requirements.remove("-e .") 
    return requirements
setup(
    name="gpu-kernel-performance-optimizer",
    version="0.0.1",
    author="Nitya",
    author_email="nityasribalaji@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)