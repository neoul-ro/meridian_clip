from setuptools import setup

package_name = 'meridian_clip'

setup(
    name=package_name,
    version='0.0.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='blu-y',
    maintainer_email='a_o@kakao.com',
    description='CLIP placeholder embedding node for the Meridian perception pipeline',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'clip_node = meridian_clip.clip_node:main',
        ],
    },
)
