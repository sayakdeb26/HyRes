from setuptools import setup

package_name = 'ui_kiosk_pkg'

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
    maintainer='sayak',
    maintainer_email='sayak@example.com',
    description='UI Kiosk Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    package_data={
        'ui_kiosk_pkg': ['www/*', 'www/static/*'],
    },
    entry_points={
        'console_scripts': [
            'ui_kiosk_node = ui_kiosk_pkg.ui_kiosk_node:main',
        ],
    },
)
