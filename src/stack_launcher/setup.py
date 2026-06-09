from glob import glob

from setuptools import setup

package_name = 'stack_launcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robesafe',
    maintainer_email='robesafe@todo.todo',
    description='Launch files for the CARLA simulation stack',
    license='TODO',
    entry_points={
        'console_scripts': [],
    },
)
