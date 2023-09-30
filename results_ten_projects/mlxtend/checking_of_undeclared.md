# undeclared dependenciec
## nose
### 1.
**path**: `.repositories/mlxtend/mlxtend/text/tests/test_generalize_names.py`
**line number**: 4
```python
if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest

    raise SkipTest

from mlxtend.text import generalize_names


```
### 2.
**path**: `.repositories/mlxtend/mlxtend/text/tests/test_generalize_names_duplcheck.py`
**line number**: 4
```python
if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest

    raise SkipTest

from io import StringIO


```
### 3.
**path**: `.repositories/mlxtend/mlxtend/evaluate/permutation.py`
**line number**: 15
```python
try:
    from nose.tools import nottest
except ImportError:
    # Use a no-op decorator if nose is not available
    def nottest(f):
        return f


```
## packaging
### 1.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 13
```python
import pytest
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn import datasets, exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

```
### 2.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 10
```python
from numpy.testing import assert_almost_equal
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

```
### 3.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 11
```python
import pandas as pd
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

```
### 4.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 10
```python
from numpy.testing import assert_almost_equal
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

```
### 5.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 12
```python
from numpy.testing import assert_almost_equal
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

```
### 6.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 14
```python
import pytest
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

```
### 7.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 9
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 8.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 10
```python
from numpy.testing import assert_almost_equal
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

```
### 9.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 10
```python
import numpy as np
from packaging.version import Version
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

```
### 10.
**path**: `.repositories/mlxtend/mlxtend/text/names.py`
**line number**: 15
```python

from packaging.version import Version
from pandas import __version__ as pandas_version

if sys.version_info <= (3, 0):
    raise ImportError(
        "Sorry, the text.names module is incompatible"

```
### 11.
**path**: `.repositories/mlxtend/mlxtend/frequent_patterns/tests/test_fpbase.py`
**line number**: 14
```python
from numpy.testing import assert_array_equal
from packaging.version import Version
from pandas import __version__ as pandas_version
from scipy.sparse import csr_matrix

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.utils import assert_raises

```
### 12.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_pca_corr_graph.py`
**line number**: 2
```python
import pytest
from packaging.version import Version
from sklearn import __version__ as sklearn_version

from mlxtend.data import iris_data
from mlxtend.plotting import plot_pca_correlation_graph


```
### 13.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 7
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 14.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 8
```python
import numpy as np
from packaging.version import Version
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

```
### 15.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 11
```python
import pytest
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

```
### 16.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 7
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 17.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_holdout.py`
**line number**: 8
```python
import numpy as np
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import iris_data

```
## psutil
### 1.
**path**: `.repositories/mlxtend/mlxtend/externals/pyprind/prog_class.py`
**line number**: 20
```python
try:
    import psutil

    psutil_import = True
except ImportError:
    psutil_import = False


```
## tensorflow
### 1.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 163
```python
def test_keras_fitparams():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(32, activation=tf.nn.relu), tf.keras.layers.Dense(1)]
    )


```
### 2.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bias_variance_decomp.py`
**line number**: 118
```python
def test_keras():
    import tensorflow as tf

    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True
    )

```
