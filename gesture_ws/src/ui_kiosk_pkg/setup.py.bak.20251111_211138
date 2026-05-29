from setuptools import setup
package_name = 'ui_kiosk_pkg'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # Python package folder
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/ui_kiosk_pkg']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ui_kiosk.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Sayak',
    maintainer_email='sayak.deb26@gmail.com',
    description='UI kiosk node for user confirmation of VLM predictions.',
    license='MIT',
    entry_points={'console_scripts': [
        'ui_kiosk_node = ui_kiosk_pkg.ui_kiosk_node:main',
    ]},
)
