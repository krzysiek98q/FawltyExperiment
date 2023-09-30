# undeclared dependenciec
## Cython
### 1.
**path**: `.repositories/scipy/scipy/_build_utils/tempita.py`
**line number**: 5
```python

from Cython import Tempita as tempita
# XXX: If this import ever fails (does it really?), vendor either
# cython.tempita or numpy/npy_tempita.


def process_tempita(fromfile, outfile=None):

```
## cffi
### 1.
**path**: `.repositories/scipy/scipy/_lib/tests/test_ccallback.py`
**line number**: 13
```python
try:
    import cffi
    HAVE_CFFI = True
except ImportError:
    HAVE_CFFI = False



```
### 2.
**path**: `.repositories/scipy/scipy/_lib/_ccallback.py`
**line number**: 19
```python
    try:
        import cffi
        ffi = cffi.FFI()
        CData = ffi.CData
    except ImportError:
        ffi = False


```
## cupy
### 1.
**path**: `.repositories/scipy/scipy/conftest.py`
**line number**: 121
```python
    try:
        import cupy  # type: ignore[import]
        xp_available_backends.update({'cupy': cupy})
    except ImportError:
        pass

    # by default, use all available backends

```
## cupyx
### 1.
**path**: `.repositories/scipy/scipy/special/_support_alternative_backends.py`
**line number**: 24
```python
    elif is_cupy(xp):
        import cupyx  # type: ignore[import]
        f = getattr(cupyx.scipy.special, f_name, None)
    elif xp.__name__ == f"{array_api_compat_prefix}.jax":
        f = getattr(xp.scipy.special, f_name, None)
    else:
        f_scipy = getattr(_ufuncs, f_name, None)

```
## hypothesis
### 1.
**path**: `.repositories/scipy/scipy/conftest.py`
**line number**: 12
```python
import pytest
import hypothesis

from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib import _pep440
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE

```
### 2.
**path**: `.repositories/scipy/scipy/stats/tests/test_stats.py`
**line number**: 14
```python
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib

from numpy.testing import (assert_, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,

```
### 3.
**path**: `.repositories/scipy/scipy/stats/tests/test_stats.py`
**line number**: 15
```python
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib

from numpy.testing import (assert_, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_approx_equal,

```
### 4.
**path**: `.repositories/scipy/scipy/_lib/tests/test__util.py`
**line number**: 11
```python
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa
from scipy.conftest import array_api_compatible

from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,

```
### 5.
**path**: `.repositories/scipy/scipy/_lib/tests/test__util.py`
**line number**: 12
```python
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa
from scipy.conftest import array_api_compatible

from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
                              getfullargspec_no_self, FullArgSpec,

```
### 6.
**path**: `.repositories/scipy/scipy/special/tests/test_support_alternative_backends.py`
**line number**: 2
```python
import pytest
from hypothesis import given, strategies, reproduce_failure  # noqa
import hypothesis.extra.numpy as npst

from scipy.special._support_alternative_backends import (get_array_special_func,
                                                         array_special_func_map)
from scipy.conftest import array_api_compatible

```
### 7.
**path**: `.repositories/scipy/scipy/special/tests/test_support_alternative_backends.py`
**line number**: 3
```python
from hypothesis import given, strategies, reproduce_failure  # noqa
import hypothesis.extra.numpy as npst

from scipy.special._support_alternative_backends import (get_array_special_func,
                                                         array_special_func_map)
from scipy.conftest import array_api_compatible
from scipy import special

```
## packaging
### 1.
**path**: `.repositories/scipy/scipy/_lib/_testutils.py`
**line number**: 20
```python
    # Note that packaging is not a dependency, hence we need this try-except:
    from packaging.tags import sys_tags
    _tags = list(sys_tags())
    if 'musllinux' in _tags[0].platform:
        IS_MUSL = True
except ImportError:
    # fallback to sysconfig (might be flaky)

```
## platformdirs
### 1.
**path**: `.repositories/scipy/scipy/datasets/_utils.py`
**line number**: 6
```python
try:
    import platformdirs
except ImportError:
    platformdirs = None  # type: ignore[assignment]


def _clear_cache(datasets, cache_dir=None, method_map=None):

```
## psutil
### 1.
**path**: `.repositories/scipy/scipy/spatial/tests/test_hausdorff.py`
**line number**: 160
```python
    try:
        import psutil
    except ModuleNotFoundError:
        pytest.skip("psutil required to check available memory")
    if psutil.virtual_memory().available < 80*2**30:
        # Don't run the test if there is less than 80 gig of RAM available.
        pytest.skip('insufficient memory available to run this test')

```
### 2.
**path**: `.repositories/scipy/scipy/_lib/_testutils.py`
**line number**: 239
```python
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):

```
## pybind11
### 1.
**path**: `.repositories/scipy/scipy/optimize/setup.py`
**line number**: 34
```python
    from distutils.sysconfig import get_python_inc
    import pybind11

    config = Configuration('optimize', parent_package, top_path)

    include_dirs = [join(os.path.dirname(__file__), '..', '_lib', 'src')]


```
### 2.
**path**: `.repositories/scipy/scipy/io/_fast_matrix_market/setup.py`
**line number**: 3
```python

import pybind11.setup_helpers
from pybind11.setup_helpers import Pybind11Extension


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

```
### 3.
**path**: `.repositories/scipy/scipy/io/_fast_matrix_market/setup.py`
**line number**: 4
```python
import pybind11.setup_helpers
from pybind11.setup_helpers import Pybind11Extension


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_fast_matrix_market', parent_package, top_path)

```
### 4.
**path**: `.repositories/scipy/scipy/spatial/setup.py`
**line number**: 26
```python
    from distutils.sysconfig import get_python_inc
    import pybind11

    config = Configuration('spatial', parent_package, top_path)

    config.add_data_dir('tests')


```
### 5.
**path**: `.repositories/scipy/scipy/fft/_pocketfft/setup.py`
**line number**: 30
```python
    from numpy.distutils.misc_util import Configuration
    import pybind11
    include_dirs = [pybind11.get_include(True), pybind11.get_include(False)]

    config = Configuration('_pocketfft', parent_package, top_path)
    ext = config.add_extension('pypocketfft',
                               sources=['pypocketfft.cxx'],

```
## pythran
### 1.
**path**: `.repositories/scipy/scipy/optimize/setup.py`
**line number**: 132
```python
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.optimize._group_columns',
            sources=["scipy/optimize/_group_columns.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)

```
### 2.
**path**: `.repositories/scipy/scipy/linalg/setup.py`
**line number**: 110
```python
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.linalg._matfuncs_sqrtm_triu',
            sources=["scipy/linalg/_matfuncs_sqrtm_triu.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)

```
### 3.
**path**: `.repositories/scipy/scipy/signal/setup.py`
**line number**: 32
```python
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.signal._max_len_seq_inner',
            sources=["scipy/signal/_max_len_seq_inner.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)

```
### 4.
**path**: `.repositories/scipy/scipy/stats/setup.py`
**line number**: 45
```python
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.stats._stats_pythran',
            sources=["scipy/stats/_stats_pythran.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)

```
### 5.
**path**: `.repositories/scipy/scipy/interpolate/setup.py`
**line number**: 61
```python
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        from pythran.dist import PythranExtension
        ext = PythranExtension(
            'scipy.interpolate._rbfinterp_pythran',
            sources=['scipy/interpolate/_rbfinterp_pythran.py'],
            config=['compiler.blas=none']
            )

```
## scikits
### 1.
**path**: `.repositories/scipy/scipy/sparse/linalg/_dsolve/tests/test_linsolve.py`
**line number**: 33
```python
try:
    import scikits.umfpack as umfpack
    has_umfpack = True
except ImportError:
    has_umfpack = False

def toarray(a):

```
### 2.
**path**: `.repositories/scipy/scipy/optimize/_linprog_ip.py`
**line number**: 37
```python
try:
    import scikits.umfpack  # test whether to use factorized
except ImportError:
    has_umfpack = False


def _get_solver(M, sparse=False, lstsq=False, sym_pos=True,

```
### 3.
**path**: `.repositories/scipy/scipy/sparse/linalg/_dsolve/linsolve.py`
**line number**: 15
```python
try:
    import scikits.umfpack as umfpack
except ImportError:
    noScikit = True

useUmfpack = not noScikit


```
### 4.
**path**: `.repositories/scipy/scipy/optimize/tests/test_linprog.py`
**line number**: 21
```python
try:
    from scikits.umfpack import UmfpackWarning
except ImportError:
    has_umfpack = False

has_cholmod = True
try:

```
## sksparse
### 1.
**path**: `.repositories/scipy/scipy/optimize/_trustregion_constr/tests/test_projections.py`
**line number**: 10
```python
try:
    from sksparse.cholmod import cholesky_AAt  # noqa: F401
    sksparse_available = True
    available_sparse_methods = ("NormalEquation", "AugmentedSystem")
except ImportError:
    sksparse_available = False
    available_sparse_methods = ("AugmentedSystem",)

```
### 2.
**path**: `.repositories/scipy/scipy/optimize/_linprog_ip.py`
**line number**: 31
```python
try:
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:

```
### 3.
**path**: `.repositories/scipy/scipy/optimize/_linprog_ip.py`
**line number**: 32
```python
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack  # test whether to use factorized

```
### 4.
**path**: `.repositories/scipy/scipy/optimize/_linprog_ip.py`
**line number**: 33
```python
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack  # test whether to use factorized
except ImportError:

```
### 5.
**path**: `.repositories/scipy/scipy/optimize/_trustregion_constr/projections.py`
**line number**: 8
```python
try:
    from sksparse.cholmod import cholesky_AAt
    sksparse_available = True
except ImportError:
    import warnings
    sksparse_available = False
import numpy as np

```
### 6.
**path**: `.repositories/scipy/scipy/optimize/tests/test_linprog.py`
**line number**: 27
```python
try:
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod  # noqa: F401
except ImportError:
    has_cholmod = False



```
### 7.
**path**: `.repositories/scipy/scipy/optimize/tests/test_linprog.py`
**line number**: 28
```python
    import sksparse  # noqa: F401
    from sksparse.cholmod import cholesky as cholmod  # noqa: F401
except ImportError:
    has_cholmod = False


def _assert_iteration_limit_reached(res, maxiter):

```
## sympy
### 1.
**path**: `.repositories/scipy/scipy/special/tests/test_precompute_expn_asy.py`
**line number**: 7
```python
try:
    import sympy
    from sympy import Poly
except ImportError:
    sympy = MissingModule("sympy")



```
### 2.
**path**: `.repositories/scipy/scipy/special/tests/test_precompute_expn_asy.py`
**line number**: 8
```python
    import sympy
    from sympy import Poly
except ImportError:
    sympy = MissingModule("sympy")


@check_version(sympy, "1.0")

```
### 3.
**path**: `.repositories/scipy/scipy/special/_precompute/utils.py`
**line number**: 7
```python
try:
    from sympy.abc import x
except ImportError:
    pass


def lagrange_inversion(a):

```
### 4.
**path**: `.repositories/scipy/scipy/special/tests/test_precompute_gammainc.py`
**line number**: 11
```python
try:
    import sympy
except ImportError:
    sympy = MissingModule('sympy')

try:
    import mpmath as mp

```
### 5.
**path**: `.repositories/scipy/scipy/special/_precompute/wright_bessel.py`
**line number**: 13
```python
try:
    import sympy
    from sympy import EulerGamma, Rational, S, Sum, \
        factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass

```
### 6.
**path**: `.repositories/scipy/scipy/special/_precompute/wright_bessel.py`
**line number**: 14
```python
    import sympy
    from sympy import EulerGamma, Rational, S, Sum, \
        factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass


```
### 7.
**path**: `.repositories/scipy/scipy/special/_precompute/wright_bessel.py`
**line number**: 16
```python
        factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass


def series_small_a():

```
### 8.
**path**: `.repositories/scipy/scipy/interpolate/_interpnd_info.py`
**line number**: 6
```python
"""
from sympy import symbols, binomial, Matrix


def _estimate_gradients_2d_global():

    #

```
### 9.
**path**: `.repositories/scipy/scipy/special/_precompute/expn_asy.py`
**line number**: 13
```python
try:
    import sympy
    from sympy import Poly
    x = sympy.symbols('x')
except ImportError:
    pass


```
### 10.
**path**: `.repositories/scipy/scipy/special/_precompute/expn_asy.py`
**line number**: 14
```python
    import sympy
    from sympy import Poly
    x = sympy.symbols('x')
except ImportError:
    pass



```
### 11.
**path**: `.repositories/scipy/scipy/special/tests/test_precompute_utils.py`
**line number**: 8
```python
try:
    import sympy
except ImportError:
    sympy = MissingModule('sympy')

try:
    import mpmath as mp

```
## torch
### 1.
**path**: `.repositories/scipy/scipy/conftest.py`
**line number**: 113
```python
    try:
        import torch  # type: ignore[import]
        xp_available_backends.update({'pytorch': torch})
        # can use `mps` or `cpu`
        torch.set_default_device(SCIPY_DEVICE)
    except ImportError:
        pass

```
