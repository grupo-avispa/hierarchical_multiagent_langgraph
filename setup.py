from setuptools import find_packages, setup

package_name = 'hierarchical_multiagent_langgraph'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'jinja2',
    ],
    zip_safe=True,
    maintainer='Alberto Tudela',
    maintainer_email='ajtudela@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node = ' + package_name + '.main:main',
        ],
    },
)
