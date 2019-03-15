from setuptools import setup
from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rpsr-dqn',
      license='MIT',
      packages=['rpsr-dqn', 'rpsr-dqn.envs', 'rpsr-dqn.explore', 'rpsr-dqn.filters', 'rpsr-dqn.policy_opt', 'rpsr-dqn.rpspnets',
                'rpsr-dqn.rpspnets.psr_lite','rpsr-dqn.rpspnets.psr_lite.utils', 'rpsr-dqn.run',
                'rpsr-dqn.run.test_utils', 'rpsr-dqn.policy'],
      install_requires=[
          'markdown',
      ],
      package_data={'libmkl_rt': ['rpsr-dqn.rpspnets.psr_lite.utils.libmkl_rt.so']},
      include_package_data=True,
      zip_safe=False)