from setuptools import setup, find_packages

package_name = 'needle_planner'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(include=['needle_planner', 'needle_planner.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/default_params.yaml'
        ]),
        ('share/' + package_name + '/config/planner_configs', [
            'config/planner_configs/geometric.yaml',
            'config/planner_configs/hybrid.yaml',
            'config/planner_configs/linear_regression.yaml',
            'config/planner_configs/neural_network.yaml'
        ]),
        ('share/' + package_name + '/launch', [
            'launch/needle_planner.launch.py'
        ])
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.0',
        'torch>=1.9.0',
        'PyNiteFEA>=0.0.8',
        'pyyaml>=5.4.0'
    ],
    zip_safe=True,
    maintainer='Shashank Goyal',
    maintainer_email='sgoyal18@jhu.edu',
    description='ROS2 package for needle path planning using various approaches',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = needle_planner.planner_node:main',
            'geometric_planner = needle_planner.planners.geometric_planner:main',
            'neural_network_planner = needle_planner.planners.neural_network_planner:main',
            'hybrid_planner = needle_planner.planners.hybrid_planner:main',
            'sampling_planner = needle_planner.planners.sampling_planner:main',
            'linear_regression_planner = needle_planner.planners.linear_regression_planner:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ]
)