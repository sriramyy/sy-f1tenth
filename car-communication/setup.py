from setuptools import setup

package_name = 'car_communication'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='f1tenth',
    maintainer_email='f1tenth@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sender = car_communication.sender_node:main',
            'receiver = car_communication.receiver_node:main',

            'odom_sender = car_communication.odom_sender:main',
            'odom_sender = car_communication.odom_receiver:main',
        ],
    },
)