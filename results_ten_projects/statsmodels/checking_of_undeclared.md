# undeclared dependenciec
## IPython
### 1.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 103
```python
    try:
        import IPython
        print("IPython: %s" % safe_version(IPython))
    except ImportError:
        print("IPython: Not installed")
    try:
        import jinja2

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 248
```python
    try:
        import IPython
        print("IPython: %s (%s)" % (safe_version(IPython),
                                    dirname(IPython.__file__)))
    except ImportError:
        print("IPython: Not installed")
    try:

```
## black
### 1.
**path**: `.repositories/statsmodels/statsmodels/tsa/ardl/_pss_critical_values/pss-process.py`
**line number**: 11
```python
if __name__ == "__main__":
    from black import FileMode, TargetVersion, format_file_contents
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    PATH = os.environ.get("PSS_PATH", "..")
    print(f"Processing {PATH}")

```
## cvxopt
### 1.
**path**: `.repositories/statsmodels/statsmodels/discrete/tests/test_discrete.py`
**line number**: 54
```python
try:
    import cvxopt  # noqa:F401
    has_cvxopt = True
except ImportError:
    has_cvxopt = False



```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 89
```python
    try:
        from cvxopt import info
        print("cvxopt: %s" % safe_version(info, 'version'))
    except ImportError:
        print("cvxopt: Not installed")

    try:

```
### 3.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 232
```python
    try:
        from cvxopt import info
        print("cvxopt: %s (%s)" % (safe_version(info, 'version'),
                                   dirname(info.__file__)))
    except ImportError:
        print("cvxopt: Not installed")


```
### 4.
**path**: `.repositories/statsmodels/statsmodels/base/l1_cvxopt.py`
**line number**: 56
```python
    """
    from cvxopt import solvers, matrix

    start_params = np.array(start_params).ravel('F')

    ## Extract arguments
    # k_params is total number of covariates, possibly including a leading constant.

```
### 5.
**path**: `.repositories/statsmodels/statsmodels/base/l1_cvxopt.py`
**line number**: 147
```python
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    u = x_arr[k_params:]
    # Call the numpy version

```
### 6.
**path**: `.repositories/statsmodels/statsmodels/base/l1_cvxopt.py`
**line number**: 162
```python
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    # Call the numpy version
    # The derivative just appends a vector of constants

```
### 7.
**path**: `.repositories/statsmodels/statsmodels/base/l1_cvxopt.py`
**line number**: 177
```python
    """
    from cvxopt import matrix

    I = np.eye(k_params)  # noqa:E741
    A = np.concatenate((-I, -I), axis=1)
    B = np.concatenate((I, -I), axis=1)
    C = np.concatenate((A, B), axis=0)

```
### 8.
**path**: `.repositories/statsmodels/statsmodels/base/l1_cvxopt.py`
**line number**: 194
```python
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    zh_x = np.asarray(z[0]) * hess(params)
    zero_mat = np.zeros(zh_x.shape)

```
### 9.
**path**: `.repositories/statsmodels/statsmodels/stats/_knockoff.py`
**line number**: 160
```python
    try:
        from cvxopt import solvers, matrix
    except ImportError:
        raise ValueError("SDP knockoff designs require installation of cvxopt")

    nobs, nvar = exog.shape


```
### 10.
**path**: `.repositories/statsmodels/statsmodels/stats/tests/test_knockoff.py`
**line number**: 11
```python
try:
    import cvxopt  # noqa:F401
    has_cvxopt = True
except ImportError:
    has_cvxopt = False



```
### 11.
**path**: `.repositories/statsmodels/statsmodels/discrete/discrete_model.py`
**line number**: 51
```python
try:
    import cvxopt  # noqa:F401
    have_cvxopt = True
except ImportError:
    have_cvxopt = False



```
### 12.
**path**: `.repositories/statsmodels/statsmodels/regression/tests/test_regression.py`
**line number**: 40
```python
try:
    import cvxopt  # noqa:F401

    has_cvxopt = True
except ImportError:
    has_cvxopt = False


```
### 13.
**path**: `.repositories/statsmodels/statsmodels/regression/linear_model.py`
**line number**: 1166
```python

        from cvxopt import solvers
        solvers.options["show_progress"] = False

        rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
        x = np.asarray(rslt['x']).flat
        bp = x[1:p+1]

```
### 14.
**path**: `.repositories/statsmodels/statsmodels/regression/linear_model.py`
**line number**: 1143
```python
        try:
            import cvxopt
        except ImportError:
            msg = 'sqrt_lasso fitting requires the cvxopt module'
            raise ValueError(msg)

        n = len(self.endog)

```
## dateutil
### 1.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 69
```python
    try:
        import dateutil
        print("    dateutil: %s" % safe_version(dateutil))
    except ImportError:
        print("    dateutil: not installed")

    try:

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 208
```python
    try:
        import dateutil
        print("    dateutil: %s (%s)" % (safe_version(dateutil),
                                         dirname(dateutil.__file__)))
    except ImportError:
        print("    dateutil: not installed")


```
## enthought
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/pca.py`
**line number**: 107
```python
        """
        import enthought.mayavi.mlab as M
        if clf:
            M.clf()
        z3=np.zeros(3)
        v=(self.getEigenvectors()*self.getEigenvalues())
        M.quiver3d(z3,z3,z3,v[ix],v[iy],v[iz],scale_factor=5)

```
## finance
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/try_interchange.py`
**line number**: 30
```python
import tabular as tb
from finance import msft, ibm  # hack to make it run as standalone

s = ts.time_series([1,2,3,4,5],
            dates=ts.date_array(["2001-01","2001-01",
            "2001-02","2001-03","2001-03"],freq="M"))


```
## jinja2
### 1.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 108
```python
    try:
        import jinja2
        print("    jinja2: %s" % safe_version(jinja2))
    except ImportError:
        print("    jinja2: Not installed")

    try:

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 254
```python
    try:
        import jinja2
        print("    jinja2: %s (%s)" % (safe_version(jinja2),
                                       dirname(jinja2.__file__)))
    except ImportError:
        print("    jinja2: Not installed")


```
### 3.
**path**: `.repositories/statsmodels/statsmodels/multivariate/tests/test_factor.py`
**line number**: 221
```python
    try:
        from jinja2 import Template  # noqa:F401
    except ImportError:
        return
        # TODO: separate this and do pytest.skip?

    # Old implementation that warns

```
## la
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/try_interchange.py`
**line number**: 27
```python
import scikits.timeseries as ts
import la
import pandas
import tabular as tb
from finance import msft, ibm  # hack to make it run as standalone

s = ts.time_series([1,2,3,4,5],

```
## mlabwrap
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tests/maketests_mlabwrap.py`
**line number**: 148
```python
    # import mlabwrap only when run as script
    from mlabwrap import mlab
    np.set_printoptions(precision=14, linewidth=100)
    data =  HoldIt('data')
    data.xo = xo
    data.save(filename='testsave.py', comment='generated data, divide by 1000')


```
### 2.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tests/maketests_mlabwrap.py`
**line number**: 176
```python
    # import mlabwrap only when run as script
    from mlabwrap import mlab
    res_armarep =  HoldIt('armarep')
    res_armarep.ar = np.array([1.,  -0.5, +0.8])
    res_armarep.ma = np.array([1., -0.6,  0.08])

    res_armarep.marep = mlab.garchma(-res_armarep.ar[1:], res_armarep.ma[1:], 20)

```
### 3.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tests/maketests_mlabwrap.py`
**line number**: 201
```python
if __name__ == '__main__':
    from mlabwrap import mlab

    import savedrvs
    xo = savedrvs.rvsdata.xar2
    x100 = xo[-100:]/1000.
    x1000 = xo/1000.

```
## models
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/bspline.py`
**line number**: 23
```python
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting

# Issue warning regarding heavy development status of this module
import warnings
_msg = """
The bspline code is technology preview and requires significant work

```
## nitime
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/fftarma.py`
**line number**: 526
```python

    from nitime.algorithms import LD_AR_est
    #yule_AR_est(s, order, Nfreqs)
    wnt, spdnt = LD_AR_est(rvs, 10, 512)
    plt.figure()
    print('spdnt.shape', spdnt.shape)
    _ = plt.plot(spdnt.ravel())

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/examples/try_ld_nitime.py`
**line number**: 7
```python

import nitime.utils as ut

import statsmodels.api as sm

sxx=None
order = 10

```
## numdifftools
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/examples/ex_mle_arma.py`
**line number**: 12
```python

import numdifftools as ndt

from statsmodels.sandbox import tsa
from statsmodels.tsa.arma_mle import Arma  # local import
from statsmodels.tsa.arima_process import arma_generate_sample


```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tsa/mlemodel.py`
**line number**: 16
```python
try:
    import numdifftools as ndt
except ImportError:
    pass

from statsmodels.base.model import LikelihoodModel


```
### 3.
**path**: `.repositories/statsmodels/statsmodels/sandbox/distributions/estimators.py`
**line number**: 341
```python
def hess_ndt(fun, pars, args, options):
    import numdifftools as ndt
    if not ('stepMax' in options or 'stepFix' in options):
        options['stepMax'] = 1e-5
    f = lambda params: fun(params, *args)
    h = ndt.Hessian(f, **options)
    return h(pars), h

```
## pygments
### 1.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 120
```python
    try:
        import pygments
        print("    pygments: %s" % safe_version(pygments))
    except ImportError:
        print("    pygments: Not installed")

    try:

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 268
```python
    try:
        import pygments
        print("    pygments: %s (%s)" % (safe_version(pygments),
                                         dirname(pygments.__file__)))
    except ImportError:
        print("    pygments: Not installed")


```
## pymc
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/bayesprior.py`
**line number**: 7
```python
try:
    import pymc
    pymc_installed = 1
except:
    print("pymc not imported")
    pymc_installed = 0


```
## rpy
### 1.
**path**: `.repositories/statsmodels/statsmodels/examples/example_rpy.py`
**line number**: 20
```python

from rpy import r

import statsmodels.api as sm

examples = [1, 2]


```
## rpy2
### 1.
**path**: `.repositories/statsmodels/statsmodels/tsa/vector_ar/tests/test_var.py`
**line number**: 123
```python
    import pandas.rpy.common as prp
    from rpy2.robjects import r

    r.source("tests/var.R")
    return prp.convert_robj(r["result"], use_pandas=False)



```
### 2.
**path**: `.repositories/statsmodels/statsmodels/examples/tests/test_notebooks.py`
**line number**: 31
```python
try:
    import rpy2  # noqa: F401
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

try:

```
## savedrvs
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tests/maketests_mlabwrap.py`
**line number**: 203
```python

    import savedrvs
    xo = savedrvs.rvsdata.xar2
    x100 = xo[-100:]/1000.
    x1000 = xo/1000.

    filen = 'testsavetls.py'

```
## scikits
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/try_interchange.py`
**line number**: 26
```python
import numpy as np
import scikits.timeseries as ts
import la
import pandas
import tabular as tb
from finance import msft, ibm  # hack to make it run as standalone


```
### 2.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/examples/example_var.py`
**line number**: 6
```python
import numpy as np
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tplt

import statsmodels.api as sm

data = sm.datasets.macrodata.load()

```
### 3.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/examples/example_var.py`
**line number**: 7
```python
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tplt

import statsmodels.api as sm

data = sm.datasets.macrodata.load()
data = data.data

```
### 4.
**path**: `.repositories/statsmodels/statsmodels/sandbox/tsa/try_arma_more.py`
**line number**: 23
```python
try:
    import scikits.talkbox.spectral.basic as stbs
except ImportError:
    hastalkbox = False

ar = [1., -0.7]#[1,0,0,0,0,0,0,-0.7]
ma = [1., 0.3]

```
## sklearn
### 1.
**path**: `.repositories/statsmodels/statsmodels/tsa/ardl/_pss_critical_values/pss-process.py`
**line number**: 8
```python
from scipy import stats
from sklearn.model_selection import KFold

if __name__ == "__main__":
    from black import FileMode, TargetVersion, format_file_contents
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tsa/ardl/_pss_critical_values/pss-process.py`
**line number**: 12
```python
    from black import FileMode, TargetVersion, format_file_contents
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    PATH = os.environ.get("PSS_PATH", "..")
    print(f"Processing {PATH}")


```
### 3.
**path**: `.repositories/statsmodels/statsmodels/tsa/ardl/_pss_critical_values/pss-process.py`
**line number**: 13
```python
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    PATH = os.environ.get("PSS_PATH", "..")
    print(f"Processing {PATH}")

    files = glob.glob(os.path.join(PATH, "*.npz"))

```
### 4.
**path**: `.repositories/statsmodels/statsmodels/tools/parallel.py`
**line number**: 52
```python
        except ImportError:
            from sklearn.externals.joblib import Parallel, delayed

        parallel = Parallel(n_jobs, verbose=verbose)
        my_func = delayed(func)

        if n_jobs == -1:

```
### 5.
**path**: `.repositories/statsmodels/statsmodels/examples/l1_demo/sklearn_compare.py`
**line number**: 23
```python
import numpy as np
from sklearn import linear_model

import statsmodels.api as sm

## Decide which dataset to use
# Use either spector or anes96

```
### 6.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/ex_ratereturn.py`
**line number**: 81
```python
try:
    import sklearn  # noqa:F401
except ImportError:
    has_sklearn = False
    print('sklearn not available')



```
### 7.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/ex_ratereturn.py`
**line number**: 93
```python
if has_sklearn:
    from sklearn.covariance import LedoitWolf, OAS, MCD

    lw = LedoitWolf(store_precision=False)
    lw.fit(rr, assume_centered=False)
    cov_lw = lw.covariance_
    corr_lw = cov2corr(cov_lw)

```
## sympy
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/regression/sympy_diff.py`
**line number**: 7
```python
"""
import sympy as sy


def pdf(x, mu, sigma):
    """Return the probability density function as an expression in x"""
    #x = sy.sympify(x)

```
## tabular
### 1.
**path**: `.repositories/statsmodels/statsmodels/sandbox/examples/thirdparty/try_interchange.py`
**line number**: 29
```python
import pandas
import tabular as tb
from finance import msft, ibm  # hack to make it run as standalone

s = ts.time_series([1,2,3,4,5],
            dates=ts.date_array(["2001-01","2001-01",
            "2001-02","2001-03","2001-03"],freq="M"))

```
## typing_extensions
### 1.
**path**: `.repositories/statsmodels/statsmodels/compat/python.py`
**line number**: 69
```python
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:
    from typing import Any as Literal

```
## virtualenv
### 1.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 132
```python
    try:
        import virtualenv
        print("virtualenv: %s" % safe_version(virtualenv))
    except ImportError:
        print("virtualenv: Not installed")

    print("\n")

```
### 2.
**path**: `.repositories/statsmodels/statsmodels/tools/print_version.py`
**line number**: 281
```python
    try:
        import virtualenv
        print("virtualenv: %s (%s)" % (safe_version(virtualenv),
                                       dirname(virtualenv.__file__)))
    except ImportError:
        print("virtualenv: Not installed")


```
## yapf
### 1.
**path**: `.repositories/statsmodels/statsmodels/stats/tests/results/lilliefors_critical_value_simulation.py`
**line number**: 15
```python
from scipy import stats
from yapf.yapflib.yapf_api import FormatCode

import statsmodels.api as sm

NUM_SIM = 10000000
MAX_MEMORY = 2 ** 28

```
