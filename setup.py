from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'hierarchical_multiagent_langgraph'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.json'))),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'templates'),
            glob(os.path.join('templates', '*.jinja'))),
    ],
    install_requires=[
        'setuptools',
        'fastmcp',
        'jinja2',
        'langchain-core',
        'langchain-ollama',
        'langgraph',
        'langgraph-cli[inmem]',
        'langsmith',
        'ollama',
        'python-dotenv',
    ],
    zip_safe=True,
    maintainer='Alberto Tudela',
    maintainer_email='ajtudela@gmail.com',
    description='Hierarchical multi-agent system where a Supervisor coordinates '
    'multiple Single-Purpose Agents (SPAs) to execute complex tasks.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node = ' + package_name + '.main:main',
            'atomic_mcp_server = ' + package_name + '.atomic_mcp_server:main',
        ],
    },
)
