try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
            'description': 'MNIST',
            'author': 'Lee Sharkey',
            'url': 'https://github.com/harkashark/MNIST.git',
            'download_url': 'Where to download it.',
            'author_email': 'leedsharkey@gmail.com',
            'version': '0.1',
            'install_requires': ['nose'],
            'packages': ['MNIST'],
            'scripts': [],
            'name': 'mnist_neural_net'
}

setup(**config)
