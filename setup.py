from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

USE_FLOAT = 0  # use float (1) or double (0)


eigen_dir = os.path.join('include', 'Eigen')
if not os.path.exists(eigen_dir):
    print('Downloading Eigen...')

    from glob import glob
    from urllib.request import urlretrieve
    import shutil
    import tarfile

    if not os.path.exists('include'):
        os.mkdir('include')
    eigen_url = 'http://bitbucket.org/eigen/eigen/get/3.3.3.tar.gz'
    tar_path = os.path.join('include', 'Eigen.tar.gz')
    urlretrieve(eigen_url, tar_path)
    with tarfile.open(tar_path, 'r') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, "include")
    thedir = glob(os.path.join('include', 'eigen-eigen-*'))[0]
    shutil.move(os.path.join(thedir, 'Eigen'), eigen_dir)
    print('done!')


setup(
    name = 'stochs',
    ext_modules = cythonize([Extension(
        'stochs',
        ['stochs.pyx',
         'solvers/Loss.cpp',
         'solvers/SGD.cpp',
         'solvers/MISO.cpp',
         'solvers/SAGA.cpp'],
        language='c++',
        include_dirs=[numpy.get_include(), 'include', 'solvers'],
        extra_compile_args=['-std=c++11', '-fopenmp'],
        extra_link_args=['-std=c++11', '-fopenmp', '-lglog'],
        define_macros=[('USE_FLOAT', USE_FLOAT)],
        )],
        compile_time_env={'USE_FLOAT': USE_FLOAT})
)
