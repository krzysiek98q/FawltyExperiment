# undeclared dependenciec
## datamatrix
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 351
```python
            try:
                from datamatrix import DataMatrix, convert as cnv  # noqa
            except ImportError:
                raise ValueError(
                    "Failed to convert object to pandas dataframe (DataMatrix not available)"  # noqa
                )
            else:

```
## matplotlib
### 1.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 2
```python
import pytest
import matplotlib
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase

```
### 2.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 6
```python
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
from pingouin import read_dataset
from pingouin.plotting import (
    plot_blandaltman,
    _ppoints,

```
### 3.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 11
```python
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85
# sns.set(style='ticks', context='notebook')

```
### 4.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 12
```python
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85
# sns.set(style='ticks', context='notebook')


```
### 5.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 1090
```python
    """
    from matplotlib.patches import Circle
    from .circular import circ_r, circ_mean

    # Sanity checks
    angles = np.asarray(angles)
    assert angles.ndim == 1, "angles must be a one-dimensional array."

```
## mpmath
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 425
```python
    try:
        import mpmath  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :

```
### 2.
**path**: `.repositories/pingouin/pingouin/bayesian.py`
**line number**: 325
```python
            _is_mpmath_installed(raise_error=True)
            from mpmath import hyp3f2

            hyper_term = float(hyp3f2(1, n / 2, n / 2, 3 / 2, (2 + k * (n + 1)) / (2 * k), r**2))
            log_term = 2 * (lgamma(n / 2) - lgamma((n - 1) / 2)) - lbeta
            C = 2 ** ((3 * k - 2) / k) * k * r / (2 + (n - 1) * k) * exp(log_term) * hyper_term


```
## numpy
### 1.
**path**: `.repositories/pingouin/pingouin/circular.py`
**line number**: 8
```python
# Journal of Statistical Software, Articles 31 (10): 1–21.
import numpy as np
from scipy.stats import norm

from .utils import remove_na

__all__ = [

```
### 2.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 3
```python
import numbers
import numpy as np
import pandas as pd
import itertools as it
import collections.abc
from tabulate import tabulate
from .config import options

```
### 3.
**path**: `.repositories/pingouin/pingouin/contingency.py`
**line number**: 3
```python
import warnings
import numpy as np
import pandas as pd

from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence, binom, chi2 as sp_chi2


```
### 4.
**path**: `.repositories/pingouin/pingouin/nonparametric.py`
**line number**: 4
```python
import scipy
import numpy as np
import pandas as pd
from pingouin import remove_na, _check_dataframe, _postprocess_dataframe

__all__ = [
    "mad",

```
### 5.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 3
```python
# Date: April 2018
import numpy as np
import pandas as pd
import pandas_flavor as pf
from itertools import combinations, product
from pingouin.config import options
from pingouin.parametric import anova

```
### 6.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 719
```python
    from itertools import combinations
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import ttest_ind, ttest_rel

    assert isinstance(pval_stars, dict), "pval_stars must be a dictionary."
    assert isinstance(decimals, int), "decimals must be an int."

```
### 7.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 720
```python
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import ttest_ind, ttest_rel

    assert isinstance(pval_stars, dict), "pval_stars must be a dictionary."
    assert isinstance(decimals, int), "decimals must be an int."


```
### 8.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from scipy.stats import pearsonr, pointbiserialr

from pingouin.effsize import compute_esci, compute_effsize, compute_effsize_from_t, compute_bootci

```
### 9.
**path**: `.repositories/pingouin/pingouin/tests/test_multivariate.py`
**line number**: 1
```python
import numpy as np
import pandas as pd
from sklearn import datasets
from unittest import TestCase
from pingouin import read_dataset
from pingouin.multivariate import multivariate_normality, multivariate_ttest, box_m

```
### 10.
**path**: `.repositories/pingouin/pingouin/tests/test_utils.py`
**line number**: 2
```python
import pandas as pd
import numpy as np
import pytest

import pingouin

from unittest import TestCase

```
### 11.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 3
```python
import matplotlib
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
from pingouin import read_dataset

```
### 12.
**path**: `.repositories/pingouin/pingouin/tests/test_pandas.py`
**line number**: 7
```python
"""
import numpy as np
import pingouin as pg
from unittest import TestCase

df = pg.read_dataset("mixed_anova")
df_aov3 = pg.read_dataset("anova3_unbalanced")

```
### 13.
**path**: `.repositories/pingouin/pingouin/tests/test_equivalence.py`
**line number**: 3
```python
# Date July 2019
import numpy as np
from unittest import TestCase
from pingouin.equivalence import tost


class TestEquivalence(TestCase):

```
### 14.
**path**: `.repositories/pingouin/pingouin/tests/test_multicomp.py`
**line number**: 2
```python
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from unittest import TestCase
from pingouin.multicomp import fdr, bonf, holm, sidak, multicomp

pvals = [0.52, 0.12, 0.0001, 0.03, 0.14]

```
### 15.
**path**: `.repositories/pingouin/pingouin/tests/test_multicomp.py`
**line number**: 3
```python
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from unittest import TestCase
from pingouin.multicomp import fdr, bonf, holm, sidak, multicomp

pvals = [0.52, 0.12, 0.0001, 0.03, 0.14]
pvals2 = [0.52, 0.12, 0.10, 0.30, 0.14]

```
### 16.
**path**: `.repositories/pingouin/pingouin/power.py`
**line number**: 4
```python
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import brenth

__all__ = [
    "power_ttest",

```
### 17.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 3
```python
import warnings
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau


```
### 18.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 1077
```python
    """
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import pearsonr, spearmanr

    # Safety check
    assert isinstance(pval_stars, dict), "pval_stars must be a dictionnary."

```
### 19.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 1078
```python
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import pearsonr, spearmanr

    # Safety check
    assert isinstance(pval_stars, dict), "pval_stars must be a dictionnary."
    assert isinstance(decimals, int), "decimals must be an int."

```
### 20.
**path**: `.repositories/pingouin/pingouin/tests/test_pairwise.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin import read_dataset
from pingouin.pairwise import (
    pairwise_ttests,

```
### 21.
**path**: `.repositories/pingouin/pingouin/tests/test_pairwise.py`
**line number**: 697
```python
        # MultiIndex columns
        from numpy.random import random as rdm

        # Create MultiIndex dataframe
        columns = pd.MultiIndex.from_tuples(
            [
                ("Behavior", "Rating"),

```
### 22.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase

from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression

```
### 23.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 11
```python
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal

from pingouin import read_dataset
from pingouin.regression import (
    linear_regression,
    logistic_regression,

```
### 24.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 4
```python
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy.stats import f
import pandas_flavor as pf
from pingouin import (
    _check_dataframe,

```
### 25.
**path**: `.repositories/pingouin/pingouin/tests/test_circular.py`
**line number**: 2
```python
import pytest
import numpy as np
from unittest import TestCase
from scipy.stats import circmean
from pingouin import read_dataset
from pingouin.circular import convert_angles, _checkangles
from pingouin.circular import (

```
### 26.
**path**: `.repositories/pingouin/pingouin/tests/test_nonparametric.py`
**line number**: 3
```python
import scipy
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (
    mad,
    madmedianrule,

```
### 27.
**path**: `.repositories/pingouin/pingouin/equivalence.py`
**line number**: 3
```python
# Date: July 2019
import numpy as np
import pandas as pd
from pingouin.parametric import ttest
from pingouin.utils import _postprocess_dataframe



```
### 28.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 4
```python
import warnings
import numpy as np
from scipy.stats import pearsonr
from pingouin.utils import _check_eftype, remove_na

# from pingouin.distribution import homoscedasticity


```
### 29.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 7
```python
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

```
### 30.
**path**: `.repositories/pingouin/pingouin/tests/test_bayesian.py`
**line number**: 1
```python
import numpy as np
from unittest import TestCase
from scipy.stats import pearsonr
from pingouin.parametric import ttest
from pingouin.bayesian import bayesfactor_ttest, bayesfactor_binom
from pingouin.bayesian import bayesfactor_pearson as bfp

```
### 31.
**path**: `.repositories/pingouin/pingouin/tests/test_contingency.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
import pingouin as pg
from unittest import TestCase
from scipy.stats import chi2_contingency


```
### 32.
**path**: `.repositories/pingouin/pingouin/tests/test_power.py`
**line number**: 2
```python
import pytest
import numpy as np
from unittest import TestCase
from pingouin.power import (
    power_ttest,
    power_ttest2n,
    power_anova,

```
### 33.
**path**: `.repositories/pingouin/pingouin/multivariate.py`
**line number**: 1
```python
import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import remove_na, _postprocess_dataframe

__all__ = ["multivariate_normality", "multivariate_ttest", "box_m"]

```
### 34.
**path**: `.repositories/pingouin/pingouin/tests/test_reliability.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.reliability import cronbach_alpha, intraclass_corr
from pingouin import read_dataset


```
### 35.
**path**: `.repositories/pingouin/pingouin/multicomp.py`
**line number**: 3
```python
# Date: April 2018
import numpy as np
from pandas import Series

__all__ = ["multicomp"]



```
### 36.
**path**: `.repositories/pingouin/pingouin/tests/test_correlation.py`
**line number**: 2
```python
import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import corr, rm_corr, partial_corr, skipped, distance_corr, bicor
from pingouin import read_dataset



```
### 37.
**path**: `.repositories/pingouin/pingouin/tests/test_parametric.py`
**line number**: 2
```python
import pytest
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal as array_equal

from pingouin import read_dataset
from pingouin.parametric import ttest, anova, rm_anova, mixed_anova, ancova, welch_anova

```
### 38.
**path**: `.repositories/pingouin/pingouin/tests/test_parametric.py`
**line number**: 4
```python
from unittest import TestCase
from numpy.testing import assert_array_equal as array_equal

from pingouin import read_dataset
from pingouin.parametric import ttest, anova, rm_anova, mixed_anova, ancova, welch_anova



```
### 39.
**path**: `.repositories/pingouin/pingouin/distribution.py`
**line number**: 3
```python
import scipy.stats
import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import _flatten_list as _fl
from pingouin.utils import remove_na, _postprocess_dataframe


```
### 40.
**path**: `.repositories/pingouin/pingouin/bayesian.py`
**line number**: 3
```python
import warnings
import numpy as np
from scipy.integrate import quad
from math import pi, exp, log, lgamma

__all__ = ["bayesfactor_ttest", "bayesfactor_pearson", "bayesfactor_binom"]


```
### 41.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 3
```python
import warnings
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq


```
### 42.
**path**: `.repositories/pingouin/pingouin/reliability.py`
**line number**: 1
```python
import numpy as np
import pandas as pd
from scipy.stats import f
from pingouin.config import options
from pingouin.utils import _postprocess_dataframe


```
### 43.
**path**: `.repositories/pingouin/pingouin/tests/test_distribution.py`
**line number**: 2
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.distribution import (
    gzscore,
    normality,

```
## outdated
### 1.
**path**: `.repositories/pingouin/pingouin/__init__.py`
**line number**: 26
```python
# Warn if a newer version of Pingouin is available
from outdated import warn_if_outdated

warn_if_outdated("pingouin", __version__)

# load default options
set_default_options()

```
## pandas
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
import itertools as it
import collections.abc
from tabulate import tabulate
from .config import options


```
### 2.
**path**: `.repositories/pingouin/pingouin/contingency.py`
**line number**: 4
```python
import numpy as np
import pandas as pd

from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence, binom, chi2 as sp_chi2

from pingouin import power_chi2, _postprocess_dataframe

```
### 3.
**path**: `.repositories/pingouin/pingouin/nonparametric.py`
**line number**: 5
```python
import numpy as np
import pandas as pd
from pingouin import remove_na, _check_dataframe, _postprocess_dataframe

__all__ = [
    "mad",
    "madmedianrule",

```
### 4.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
import pandas_flavor as pf
from itertools import combinations, product
from pingouin.config import options
from pingouin.parametric import anova
from pingouin.multicomp import multicomp

```
### 5.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
from unittest import TestCase
from scipy.stats import pearsonr, pointbiserialr

from pingouin.effsize import compute_esci, compute_effsize, compute_effsize_from_t, compute_bootci
from pingouin.effsize import convert_effsize as cef

```
### 6.
**path**: `.repositories/pingouin/pingouin/tests/test_multivariate.py`
**line number**: 2
```python
import numpy as np
import pandas as pd
from sklearn import datasets
from unittest import TestCase
from pingouin import read_dataset
from pingouin.multivariate import multivariate_normality, multivariate_ttest, box_m


```
### 7.
**path**: `.repositories/pingouin/pingouin/tests/test_utils.py`
**line number**: 1
```python
import pandas as pd
import numpy as np
import pytest

import pingouin


```
### 8.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

from pingouin.config import options

```
### 9.
**path**: `.repositories/pingouin/pingouin/tests/test_pairwise.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin import read_dataset
from pingouin.pairwise import (
    pairwise_ttests,
    pairwise_tests,

```
### 10.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
from unittest import TestCase

from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

```
### 11.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 10
```python

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal

from pingouin import read_dataset
from pingouin.regression import (
    linear_regression,

```
### 12.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 5
```python
import numpy as np
import pandas as pd
from scipy.stats import f
import pandas_flavor as pf
from pingouin import (
    _check_dataframe,
    remove_na,

```
### 13.
**path**: `.repositories/pingouin/pingouin/tests/test_nonparametric.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (
    mad,
    madmedianrule,
    mwu,

```
### 14.
**path**: `.repositories/pingouin/pingouin/equivalence.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
from pingouin.parametric import ttest
from pingouin.utils import _postprocess_dataframe


__all__ = ["tost"]

```
### 15.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 8
```python
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


```
### 16.
**path**: `.repositories/pingouin/pingouin/tests/test_contingency.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
import pingouin as pg
from unittest import TestCase
from scipy.stats import chi2_contingency

df_ind = pg.read_dataset("chi2_independence")

```
### 17.
**path**: `.repositories/pingouin/pingouin/multivariate.py`
**line number**: 2
```python
import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import remove_na, _postprocess_dataframe

__all__ = ["multivariate_normality", "multivariate_ttest", "box_m"]


```
### 18.
**path**: `.repositories/pingouin/pingouin/datasets/__init__.py`
**line number**: 1
```python
import pandas as pd
import os.path as op
from pingouin.utils import print_table

ddir = op.dirname(op.realpath(__file__))
dts = pd.read_csv(op.join(ddir, "datasets.csv"), sep=",")

```
### 19.
**path**: `.repositories/pingouin/pingouin/tests/test_reliability.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.reliability import cronbach_alpha, intraclass_corr
from pingouin import read_dataset



```
### 20.
**path**: `.repositories/pingouin/pingouin/multicomp.py`
**line number**: 4
```python
import numpy as np
from pandas import Series

__all__ = ["multicomp"]


##############################################################################

```
### 21.
**path**: `.repositories/pingouin/pingouin/distribution.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import _flatten_list as _fl
from pingouin.utils import remove_na, _postprocess_dataframe



```
### 22.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 4
```python
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq

from pingouin.config import options

```
### 23.
**path**: `.repositories/pingouin/pingouin/reliability.py`
**line number**: 2
```python
import numpy as np
import pandas as pd
from scipy.stats import f
from pingouin.config import options
from pingouin.utils import _postprocess_dataframe



```
### 24.
**path**: `.repositories/pingouin/pingouin/tests/test_distribution.py`
**line number**: 3
```python
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.distribution import (
    gzscore,
    normality,
    anderson,

```
## pandas_flavor
### 1.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 5
```python
import pandas as pd
import pandas_flavor as pf
from itertools import combinations, product
from pingouin.config import options
from pingouin.parametric import anova
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_effsize

```
### 2.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 5
```python
import pandas as pd
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

from pingouin.config import options
from pingouin.power import power_corr

```
### 3.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 7
```python
from scipy.stats import f
import pandas_flavor as pf
from pingouin import (
    _check_dataframe,
    remove_na,
    _flatten_list,
    bayesfactor_ttest,

```
### 4.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 5
```python
import pandas as pd
import pandas_flavor as pf
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq

from pingouin.config import options
from pingouin.utils import remove_na as rm_na

```
## pytest
### 1.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from scipy.stats import pearsonr, pointbiserialr


```
### 2.
**path**: `.repositories/pingouin/pingouin/tests/test_utils.py`
**line number**: 3
```python
import numpy as np
import pytest

import pingouin

from unittest import TestCase
from pingouin.utils import (

```
### 3.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 1
```python
import pytest
import matplotlib
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

```
### 4.
**path**: `.repositories/pingouin/pingouin/tests/test_multicomp.py`
**line number**: 1
```python
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from unittest import TestCase
from pingouin.multicomp import fdr, bonf, holm, sidak, multicomp


```
### 5.
**path**: `.repositories/pingouin/pingouin/tests/test_pairwise.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin import read_dataset
from pingouin.pairwise import (

```
### 6.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase

from scipy.stats import linregress, zscore

```
### 7.
**path**: `.repositories/pingouin/pingouin/tests/test_circular.py`
**line number**: 1
```python
import pytest
import numpy as np
from unittest import TestCase
from scipy.stats import circmean
from pingouin import read_dataset
from pingouin.circular import convert_angles, _checkangles

```
### 8.
**path**: `.repositories/pingouin/pingouin/tests/test_nonparametric.py`
**line number**: 1
```python
import pytest
import scipy
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (

```
### 9.
**path**: `.repositories/pingouin/pingouin/tests/test_bayesian.py`
**line number**: 8
```python

from pytest import approx

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
z = np.random.normal(loc=0.5, size=100)

```
### 10.
**path**: `.repositories/pingouin/pingouin/tests/test_contingency.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
import pingouin as pg
from unittest import TestCase
from scipy.stats import chi2_contingency

```
### 11.
**path**: `.repositories/pingouin/pingouin/tests/test_power.py`
**line number**: 1
```python
import pytest
import numpy as np
from unittest import TestCase
from pingouin.power import (
    power_ttest,
    power_ttest2n,

```
### 12.
**path**: `.repositories/pingouin/pingouin/tests/test_reliability.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.reliability import cronbach_alpha, intraclass_corr
from pingouin import read_dataset

```
### 13.
**path**: `.repositories/pingouin/pingouin/tests/test_correlation.py`
**line number**: 1
```python
import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import corr, rm_corr, partial_corr, skipped, distance_corr, bicor
from pingouin import read_dataset


```
### 14.
**path**: `.repositories/pingouin/pingouin/tests/test_parametric.py`
**line number**: 1
```python
import pytest
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal as array_equal

from pingouin import read_dataset

```
### 15.
**path**: `.repositories/pingouin/pingouin/tests/test_distribution.py`
**line number**: 1
```python
import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.distribution import (
    gzscore,

```
## scipy
### 1.
**path**: `.repositories/pingouin/pingouin/circular.py`
**line number**: 9
```python
import numpy as np
from scipy.stats import norm

from .utils import remove_na

__all__ = [
    "convert_angles",

```
### 2.
**path**: `.repositories/pingouin/pingouin/circular.py`
**line number**: 596
```python
    """
    from scipy.stats import pearsonr, chi2

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.size == y.size, "x and y must have the same length."


```
### 3.
**path**: `.repositories/pingouin/pingouin/contingency.py`
**line number**: 6
```python

from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence, binom, chi2 as sp_chi2

from pingouin import power_chi2, _postprocess_dataframe



```
### 4.
**path**: `.repositories/pingouin/pingouin/contingency.py`
**line number**: 7
```python
from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence, binom, chi2 as sp_chi2

from pingouin import power_chi2, _postprocess_dataframe


__all__ = ["chi2_independence", "chi2_mcnemar", "dichotomous_crosstab"]

```
### 5.
**path**: `.repositories/pingouin/pingouin/nonparametric.py`
**line number**: 3
```python
# Date: May 2018
import scipy
import numpy as np
import pandas as pd
from pingouin import remove_na, _check_dataframe, _postprocess_dataframe

__all__ = [

```
### 6.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 12
```python
from pingouin.utils import _check_dataframe, _flatten_list, _postprocess_dataframe
from scipy.stats import studentized_range
import warnings

__all__ = [
    "pairwise_ttests",
    "pairwise_tests",

```
### 7.
**path**: `.repositories/pingouin/pingouin/pairwise.py`
**line number**: 721
```python
    from numpy import format_float_positional as ffp
    from scipy.stats import ttest_ind, ttest_rel

    assert isinstance(pval_stars, dict), "pval_stars must be a dictionary."
    assert isinstance(decimals, int), "decimals must be an int."

    if paired:

```
### 8.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 5
```python
from unittest import TestCase
from scipy.stats import pearsonr, pointbiserialr

from pingouin.effsize import compute_esci, compute_effsize, compute_effsize_from_t, compute_bootci
from pingouin.effsize import convert_effsize as cef

# Dataset

```
### 9.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 121
```python
        # 3. Univariate custom function: skewness
        from scipy.stats import skew

        n_boot = 10000
        ci_n = compute_bootci(x_m, func=skew, method="norm", n_boot=n_boot, decimals=1, seed=42)
        ci_p = compute_bootci(x_m, func=skew, method="per", n_boot=n_boot, decimals=1, seed=42)
        ci_c = compute_bootci(x_m, func=skew, method="cper", n_boot=n_boot, decimals=1, seed=42)

```
### 10.
**path**: `.repositories/pingouin/pingouin/tests/test_effsize.py`
**line number**: 132
```python
        # 4. Bivariate custom function: paired T-test
        from scipy.stats import ttest_rel

        ci_n = compute_bootci(
            x_m,
            y_m,
            func=lambda x, y: ttest_rel(x, y)[0],

```
### 11.
**path**: `.repositories/pingouin/pingouin/tests/test_multivariate.py`
**line number**: 96
```python
        # >>> boxM(data[, c('A', 'B', 'C')], grouping=data[, c('group')])
        from scipy.stats import multivariate_normal as mvn

        data = pd.DataFrame(mvn.rvs(size=(100, 3), random_state=42), columns=["A", "B", "C"])
        data["group"] = [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25
        stats = box_m(data, dvs=["A", "B", "C"], group="group")
        assert round(stats.at["box", "Chi2"], 5) == 11.63419

```
### 12.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 4
```python
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
from pingouin import read_dataset
from pingouin.plotting import (

```
### 13.
**path**: `.repositories/pingouin/pingouin/power.py`
**line number**: 5
```python
import numpy as np
from scipy import stats
from scipy.optimize import brenth

__all__ = [
    "power_ttest",
    "power_ttest2n",

```
### 14.
**path**: `.repositories/pingouin/pingouin/power.py`
**line number**: 6
```python
from scipy import stats
from scipy.optimize import brenth

__all__ = [
    "power_ttest",
    "power_ttest2n",
    "power_anova",

```
### 15.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 6
```python
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

from pingouin.config import options
from pingouin.power import power_corr
from pingouin.multicomp import multicomp

```
### 16.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 7
```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

from pingouin.config import options
from pingouin.power import power_corr
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_esci

```
### 17.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 49
```python
    """
    from scipy.stats import t

    assert alternative in [
        "two-sided",
        "greater",
        "less",

```
### 18.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 123
```python
    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet

    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))

```
### 19.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 1079
```python
    from numpy import format_float_positional as ffp
    from scipy.stats import pearsonr, spearmanr

    # Safety check
    assert isinstance(pval_stars, dict), "pval_stars must be a dictionnary."
    assert isinstance(decimals, int), "decimals must be an int."
    assert method in ["pearson", "spearman"], "Method is not recognized."

```
### 20.
**path**: `.repositories/pingouin/pingouin/tests/test_pairwise.py`
**line number**: 490
```python
        from itertools import combinations
        from scipy.stats import ttest_ind, ttest_rel

        # Load BFI dataset
        df = read_dataset("pairwise_corr").iloc[:30, 1:]
        df.columns = ["N", "E", "O", "A", "C"]
        # Add some missing values

```
### 21.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 6
```python

from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal

```
### 22.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 6
```python
import pandas as pd
from scipy.stats import f
import pandas_flavor as pf
from pingouin import (
    _check_dataframe,
    remove_na,
    _flatten_list,

```
### 23.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 207
```python
    """
    from scipy.stats import t, ttest_rel, ttest_ind, ttest_1samp

    try:  # pragma: no cover
        from scipy.stats._stats_py import _unequal_var_ttest_denom, _equal_var_ttest_denom
    except ImportError:  # pragma: no cover
        # Fallback for scipy<1.8.0

```
### 24.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 210
```python
    try:  # pragma: no cover
        from scipy.stats._stats_py import _unequal_var_ttest_denom, _equal_var_ttest_denom
    except ImportError:  # pragma: no cover
        # Fallback for scipy<1.8.0
        from scipy.stats.stats import _unequal_var_ttest_denom, _equal_var_ttest_denom
    from pingouin import power_ttest, power_ttest2n, compute_effsize


```
### 25.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 213
```python
        # Fallback for scipy<1.8.0
        from scipy.stats.stats import _unequal_var_ttest_denom, _equal_var_ttest_denom
    from pingouin import power_ttest, power_ttest2n, compute_effsize

    # Check arguments
    assert alternative in [
        "two-sided",

```
### 26.
**path**: `.repositories/pingouin/pingouin/tests/test_circular.py`
**line number**: 4
```python
from unittest import TestCase
from scipy.stats import circmean
from pingouin import read_dataset
from pingouin.circular import convert_angles, _checkangles
from pingouin.circular import (
    circ_axial,
    circ_corrcc,

```
### 27.
**path**: `.repositories/pingouin/pingouin/tests/test_nonparametric.py`
**line number**: 2
```python
import pytest
import scipy
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (
    mad,

```
### 28.
**path**: `.repositories/pingouin/pingouin/tests/test_nonparametric.py`
**line number**: 32
```python
        """Test function mad."""
        from scipy.stats import median_abs_deviation as mad_scp

        a = [1.2, 3, 4.5, 2.4, 5, 6.7, 0.4]
        # Compare to Matlab
        assert mad(a, normalize=False) == 1.8
        assert np.round(mad(a), 3) == np.round(1.8 * 1.4826, 3)

```
### 29.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 5
```python
import numpy as np
from scipy.stats import pearsonr
from pingouin.utils import _check_eftype, remove_na

# from pingouin.distribution import homoscedasticity



```
### 30.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 139
```python
    """
    from scipy.stats import norm, t

    assert eftype.lower() in ["r", "pearson", "spearman", "cohen", "d", "g", "hedges"]
    assert alternative in [
        "two-sided",
        "greater",

```
### 31.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 360
```python
    from inspect import isfunction
    from scipy.stats import norm

    # Check other arguments
    assert isinstance(confidence, float)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."
    assert method in ["norm", "normal", "percentile", "per", "cpercentile", "cper"]

```
### 32.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 397
```python
        elif func == "spearman":
            from scipy.stats import spearmanr

            assert paired, "Paired should be True if using correlation functions."

            def func(x, y):
                return spearmanr(x, y)[0]

```
### 33.
**path**: `.repositories/pingouin/pingouin/effsize.py`
**line number**: 647
```python
        # Ruscio 2008
        from scipy.stats import norm

        return norm.cdf(d / np.sqrt(2))


def compute_effsize(x, y, paired=False, eftype="cohen"):

```
### 34.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 10
```python
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85

```
### 35.
**path**: `.repositories/pingouin/pingouin/tests/test_bayesian.py`
**line number**: 3
```python
from unittest import TestCase
from scipy.stats import pearsonr
from pingouin.parametric import ttest
from pingouin.bayesian import bayesfactor_ttest, bayesfactor_binom
from pingouin.bayesian import bayesfactor_pearson as bfp

from pytest import approx

```
### 36.
**path**: `.repositories/pingouin/pingouin/tests/test_contingency.py`
**line number**: 6
```python
from unittest import TestCase
from scipy.stats import chi2_contingency

df_ind = pg.read_dataset("chi2_independence")
df_mcnemar = pg.read_dataset("chi2_mcnemar")

data_ct = pd.DataFrame(

```
### 37.
**path**: `.repositories/pingouin/pingouin/multivariate.py`
**line number**: 63
```python
    """
    from scipy.stats import lognorm

    # Check input and remove missing values
    X = np.asarray(X)
    assert X.ndim == 2, "X must be of shape (n_samples, n_features)."
    X = X[~np.isnan(X).any(axis=1)]

```
### 38.
**path**: `.repositories/pingouin/pingouin/multivariate.py`
**line number**: 202
```python
    """
    from scipy.stats import f

    x = np.asarray(X)
    assert x.ndim == 2, "x must be of shape (n_samples, n_features)"

    if Y is None:

```
### 39.
**path**: `.repositories/pingouin/pingouin/multivariate.py`
**line number**: 350
```python
    # Safety checks
    from scipy.stats import chi2

    assert isinstance(data, pd.DataFrame), "data must be a pandas dataframe."
    assert group in data.columns, "The grouping variable is not in data."
    assert set(dvs).issubset(data.columns), "The DVs are not in data."
    grp = data.groupby(group, observed=True)[dvs]

```
### 40.
**path**: `.repositories/pingouin/pingouin/distribution.py`
**line number**: 2
```python
import warnings
import scipy.stats
import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import _flatten_list as _fl
from pingouin.utils import remove_na, _postprocess_dataframe

```
### 41.
**path**: `.repositories/pingouin/pingouin/bayesian.py`
**line number**: 4
```python
import numpy as np
from scipy.integrate import quad
from math import pi, exp, log, lgamma

__all__ = ["bayesfactor_ttest", "bayesfactor_pearson", "bayesfactor_binom"]



```
### 42.
**path**: `.repositories/pingouin/pingouin/bayesian.py`
**line number**: 270
```python
    """
    from scipy.special import gamma, betaln, hyp2f1

    assert method.lower() in ["ly", "wetzels"], "Method not recognized."
    assert alternative in [
        "two-sided",
        "greater",

```
### 43.
**path**: `.repositories/pingouin/pingouin/bayesian.py`
**line number**: 446
```python
    """
    from scipy.stats import beta, binom

    assert 0 < p < 1, "p must be between 0 and 1."
    assert isinstance(k, int), "k must be int."
    assert isinstance(n, int), "n must be int."
    assert k <= n, "k (successes) cannot be higher than n (trials)."

```
### 44.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 6
```python
import pandas_flavor as pf
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq

from pingouin.config import options
from pingouin.utils import remove_na as rm_na
from pingouin.utils import _flatten_list as _fl

```
### 45.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 7
```python
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq

from pingouin.config import options
from pingouin.utils import remove_na as rm_na
from pingouin.utils import _flatten_list as _fl
from pingouin.utils import _postprocess_dataframe

```
### 46.
**path**: `.repositories/pingouin/pingouin/reliability.py`
**line number**: 3
```python
import pandas as pd
from scipy.stats import f
from pingouin.config import options
from pingouin.utils import _postprocess_dataframe


__all__ = ["cronbach_alpha", "intraclass_corr"]

```
## seaborn
### 1.
**path**: `.repositories/pingouin/pingouin/tests/test_plotting.py`
**line number**: 5
```python
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
from pingouin import read_dataset
from pingouin.plotting import (
    plot_blandaltman,

```
### 2.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 9
```python
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)

```
## sklearn
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 411
```python
    try:
        import sklearn  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :

```
### 2.
**path**: `.repositories/pingouin/pingouin/tests/test_multivariate.py`
**line number**: 3
```python
import pandas as pd
from sklearn import datasets
from unittest import TestCase
from pingouin import read_dataset
from pingouin.multivariate import multivariate_normality, multivariate_ttest, box_m

data = read_dataset("multivariate")

```
### 3.
**path**: `.repositories/pingouin/pingouin/correlation.py`
**line number**: 124
```python
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet

    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    # Compute center and distance to center

```
### 4.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 7
```python
from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal


```
### 5.
**path**: `.repositories/pingouin/pingouin/regression.py`
**line number**: 829
```python
    _is_sklearn_installed(raise_error=True)
    from sklearn.linear_model import LogisticRegression

    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):

```
## statsmodels
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 397
```python
    try:
        import statsmodels  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :

```
### 2.
**path**: `.repositories/pingouin/pingouin/tests/test_regression.py`
**line number**: 8
```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal

from pingouin import read_dataset

```
### 3.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 1144
```python
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Validate the dataframe
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")
    all_cols = _flatten_list([dv, between])

```
### 4.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 1145
```python
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Validate the dataframe
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")
    all_cols = _flatten_list([dv, between])
    bad_chars = [",", "(", ")", ":"]

```
### 5.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 1689
```python
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Safety checks
    assert effsize in ["np2", "n2"], "effsize must be 'np2' or 'n2'."
    assert isinstance(data, pd.DataFrame), "data must be a pandas dataframe."

```
### 6.
**path**: `.repositories/pingouin/pingouin/parametric.py`
**line number**: 1690
```python
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Safety checks
    assert effsize in ["np2", "n2"], "effsize must be 'np2' or 'n2'."
    assert isinstance(data, pd.DataFrame), "data must be a pandas dataframe."
    assert isinstance(between, str), (

```
### 7.
**path**: `.repositories/pingouin/pingouin/plotting.py`
**line number**: 981
```python
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.formula.api import ols

    # Safety check (duplicated from pingouin.rm_corr)
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
    assert x in data.columns, "The %s column is not in data." % x
    assert y in data.columns, "The %s column is not in data." % y

```
## tabulate
### 1.
**path**: `.repositories/pingouin/pingouin/utils.py`
**line number**: 7
```python
import collections.abc
from tabulate import tabulate
from .config import options

__all__ = [
    "_perm_pval",
    "print_table",

```
