# undeclared dependenciec
## dlib
### 1.
**path**: `.repositories/mlxtend/mlxtend/image/extract_face_landmarks.py`
**line number**: 13
```python

import dlib
import numpy as np

from .utils import check_exists, download_url, extract_file

predictor_path = "~/mlxtend_data/shape_predictor_68_face_landmarks.dat"

```
## imageio
### 1.
**path**: `.repositories/mlxtend/mlxtend/image/utils.py`
**line number**: 15
```python

import imageio


def check_exists(path):
    path = os.path.expanduser(path)
    return os.path.exists(path)

```
### 2.
**path**: `.repositories/mlxtend/mlxtend/image/tests/test_eyepad_align.py`
**line number**: 9
```python

import imageio
import numpy as np
import pytest

from mlxtend.image import EyepadAlign, extract_face_landmarks


```
### 3.
**path**: `.repositories/mlxtend/mlxtend/image/tests/test_extract_face_landmarks.py`
**line number**: 9
```python

import imageio
import numpy as np
import pytest

from mlxtend.image import extract_face_landmarks


```
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
### 3.
**path**: `.repositories/mlxtend/mlxtend/text/tests/test_generalize_names_duplcheck.py`
**line number**: 4
```python
if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest

    raise SkipTest

from io import StringIO


```
## packaging
### 1.
**path**: `.repositories/mlxtend/mlxtend/text/names.py`
**line number**: 15
```python

from packaging.version import Version
from pandas import __version__ as pandas_version

if sys.version_info <= (3, 0):
    raise ImportError(
        "Sorry, the text.names module is incompatible"

```
### 2.
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
### 3.
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
### 4.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 7
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 5.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 9
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 6.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 7
```python

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


```
### 7.
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
### 8.
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
### 9.
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
### 10.
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
### 11.
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
### 12.
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
### 13.
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
### 14.
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
### 15.
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
### 16.
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
### 17.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_pca_corr_graph.py`
**line number**: 2
```python
import pytest
from packaging.version import Version
from sklearn import __version__ as sklearn_version

from mlxtend.data import iris_data
from mlxtend.plotting import plot_pca_correlation_graph


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
## six
### 1.
**path**: `.repositories/mlxtend/mlxtend/image/utils.py`
**line number**: 48
```python
def download_url(url, save_path):
    from six.moves import urllib

    save_path = os.path.expanduser(save_path)
    if not check_exists(save_path):
        makedir(save_path)


```
## skimage
### 1.
**path**: `.repositories/mlxtend/mlxtend/image/eyepad_align.py`
**line number**: 14
```python
import numpy as np
from skimage.transform import AffineTransform, resize, warp

from ..externals.pyprind.progbar import ProgBar
from . import extract_face_landmarks
from .utils import read_image


```
## sklearn
### 1.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_fixed_features.py`
**line number**: 9
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


```
### 2.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_fixed_features.py`
**line number**: 10
```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises

iris = load_iris()

```
### 3.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_softmax_regression.py`
**line number**: 8
```python
import numpy as np
from sklearn.base import clone

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 4.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_learning_curves.py`
**line number**: 8
```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_learning_curves


```
### 5.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_learning_curves.py`
**line number**: 9
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_learning_curves



```
### 6.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_learning_curves.py`
**line number**: 10
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_learning_curves


def test_training_size():

```
### 7.
**path**: `.repositories/mlxtend/mlxtend/feature_extraction/tests/test_kernel_pca.py`
**line number**: 10
```python
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons

from mlxtend.feature_extraction import RBFKernelPCA as KPCA

X1, y1 = make_moons(n_samples=50, random_state=1)


```
### 8.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_transactionencoder.py`
**line number**: 9
```python
from scipy.sparse import csr_matrix
from sklearn.base import clone

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.utils import assert_raises

dataset = [

```
### 9.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_regression.py`
**line number**: 13
```python
import scipy.sparse as sparse
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition

```
### 10.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_regression.py`
**line number**: 14
```python
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition


```
### 11.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_counterfactual.py`
**line number**: 8
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from mlxtend.classifier import OneRClassifier
from mlxtend.data import iris_data
from mlxtend.evaluate import create_counterfactual
from mlxtend.utils import assert_raises

```
### 12.
**path**: `.repositories/mlxtend/mlxtend/classifier/stacking_cv_classification.py`
**line number**: 14
```python
from scipy import sparse
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators

```
### 13.
**path**: `.repositories/mlxtend/mlxtend/classifier/stacking_cv_classification.py`
**line number**: 15
```python
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition

```
### 14.
**path**: `.repositories/mlxtend/mlxtend/classifier/stacking_cv_classification.py`
**line number**: 16
```python
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from ._base_classification import _BaseStackingClassifier

```
### 15.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 16
```python
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

```
### 16.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 17
```python
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


```
### 17.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 18
```python
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError

```
### 18.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 19
```python
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingCVRegressor

```
### 19.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 20
```python
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingCVRegressor
from mlxtend.utils import assert_raises

```
### 20.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_cv_regression.py`
**line number**: 21
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingCVRegressor
from mlxtend.utils import assert_raises


```
### 21.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/copy_transformer.py`
**line number**: 11
```python
from scipy.sparse import issparse
from sklearn.base import BaseEstimator


class CopyTransformer(BaseEstimator):
    """Transformer that returns a copy of the input array


```
### 22.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_linear_regression.py`
**line number**: 10
```python
from numpy.testing import assert_almost_equal
from sklearn.base import clone

from mlxtend.data import boston_housing_data
from mlxtend.regressor import LinearRegression

X, y = boston_housing_data()

```
### 23.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_holdout.py`
**line number**: 9
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import PredefinedHoldoutSplit, RandomHoldoutSplit

```
### 24.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_holdout.py`
**line number**: 10
```python
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import PredefinedHoldoutSplit, RandomHoldoutSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

```
### 25.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_holdout.py`
**line number**: 11
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import PredefinedHoldoutSplit, RandomHoldoutSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


```
### 26.
**path**: `.repositories/mlxtend/mlxtend/plotting/plot_linear_regression.py`
**line number**: 13
```python
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def plot_linear_regression(
    X,
    y,

```
### 27.
**path**: `.repositories/mlxtend/mlxtend/utils/base_compostion.py`
**line number**: 3
```python

from sklearn.utils.metaestimators import _BaseComposition


class _BaseXComposition(_BaseComposition):
    """
    parameter handler for list of estimators

```
### 28.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/dense_transformer.py`
**line number**: 11
```python
from scipy.sparse import issparse
from sklearn.base import BaseEstimator


class DenseTransformer(BaseEstimator):
    """
    Convert a sparse array into a dense array.

```
### 29.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/sequential_feature_selector.py`
**line number**: 19
```python
from joblib import Parallel, delayed
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess

```
### 30.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/sequential_feature_selector.py`
**line number**: 20
```python
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess


```
### 31.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 8
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data

```
### 32.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 9
```python
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import combined_ftest_5x2cv

```
### 33.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 10
```python
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import combined_ftest_5x2cv


```
### 34.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_combined_ftest_5x2cv.py`
**line number**: 11
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import combined_ftest_5x2cv



```
### 35.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 10
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data

```
### 36.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 11
```python
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_resampled

```
### 37.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 12
```python
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_resampled
from mlxtend.utils import assert_raises

```
### 38.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_resampled.py`
**line number**: 13
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_resampled
from mlxtend.utils import assert_raises


```
### 39.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 8
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data

```
### 40.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 9
```python
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_5x2cv

```
### 41.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 10
```python
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_5x2cv


```
### 42.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_5x2cv.py`
**line number**: 11
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_5x2cv



```
### 43.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 11
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score

```
### 44.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 12
```python
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold

```
### 45.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 13
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier

```
### 46.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 14
```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

```
### 47.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 15
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


```
### 48.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 16
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression

```
### 49.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 17
```python
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data

```
### 50.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 18
```python
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

```
### 51.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector.py`
**line number**: 19
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises

```
### 52.
**path**: `.repositories/mlxtend/mlxtend/evaluate/f_test.py`
**line number**: 12
```python
import scipy.stats
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split


def ftest(y_target, *y_model_predictions):
    """

```
### 53.
**path**: `.repositories/mlxtend/mlxtend/evaluate/f_test.py`
**line number**: 13
```python
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split


def ftest(y_target, *y_model_predictions):
    """
    F-Test test to compare 2 or more models.

```
### 54.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_time_series.py`
**line number**: 10
```python
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from mlxtend.evaluate import GroupTimeSeriesSplit



```
### 55.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_time_series.py`
**line number**: 11
```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from mlxtend.evaluate import GroupTimeSeriesSplit


@pytest.fixture

```
### 56.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/exhaustive_feature_selector.py`
**line number**: 21
```python
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess


```
### 57.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/exhaustive_feature_selector.py`
**line number**: 22
```python
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess



```
### 58.
**path**: `.repositories/mlxtend/mlxtend/evaluate/ttest.py`
**line number**: 10
```python
from scipy import stats
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split


def paired_ttest_resampled(
    estimator1,

```
### 59.
**path**: `.repositories/mlxtend/mlxtend/evaluate/ttest.py`
**line number**: 11
```python
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split


def paired_ttest_resampled(
    estimator1,
    estimator2,

```
### 60.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 10
```python
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

```
### 61.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 11
```python
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

```
### 62.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 12
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


```
### 63.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 13
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import DenseTransformer

```
### 64.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 14
```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import DenseTransformer


```
### 65.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 15
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import DenseTransformer

iris = load_iris()

```
### 66.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_dense_transformer.py`
**line number**: 16
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import DenseTransformer

iris = load_iris()
X, y = iris.data, iris.target

```
### 67.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_kfold.py`
**line number**: 7
```python

from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_kfold_cv

```
### 68.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_kfold.py`
**line number**: 8
```python
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_kfold_cv


```
### 69.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_paired_ttest_kfold.py`
**line number**: 9
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_kfold_cv



```
### 70.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_perceptron.py`
**line number**: 10
```python
import numpy as np
from sklearn.base import clone

from mlxtend.classifier import Perceptron
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 71.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_multilayerperceptron.py`
**line number**: 8
```python
import numpy as np
from sklearn.base import clone

from mlxtend.classifier import MultiLayerPerceptron as MLP
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 72.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_adaline.py`
**line number**: 10
```python
import numpy as np
from sklearn.base import clone

from mlxtend.classifier import Adaline
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 73.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 11
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

```
### 74.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 12
```python
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier


```
### 75.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 13
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import SoftmaxRegression

```
### 76.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 14
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data

```
### 77.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 15
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

```
### 78.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_exhaustive_feature_selector.py`
**line number**: 16
```python
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.utils import assert_raises

```
### 79.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/column_selector.py`
**line number**: 11
```python
import numpy as np
from sklearn.base import BaseEstimator


class ColumnSelector(BaseEstimator):
    """Object for selecting specific columns from a data set.


```
### 80.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 12
```python
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

```
### 81.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 13
```python
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

```
### 82.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 14
```python
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


```
### 83.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 15
```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import CopyTransformer

```
### 84.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 16
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import CopyTransformer
from mlxtend.utils import assert_raises

```
### 85.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 17
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import CopyTransformer
from mlxtend.utils import assert_raises


```
### 86.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/tests/test_copy_transformer.py`
**line number**: 18
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import CopyTransformer
from mlxtend.utils import assert_raises

iris = load_iris()

```
### 87.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bias_variance_decomp.py`
**line number**: 14
```python
import pytest
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import bias_variance_decomp

```
### 88.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bias_variance_decomp.py`
**line number**: 15
```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import bias_variance_decomp


```
### 89.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bias_variance_decomp.py`
**line number**: 16
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import bias_variance_decomp



```
### 90.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 12
```python
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import bootstrap_point632_score

```
### 91.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 13
```python
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import bootstrap_point632_score
from mlxtend.utils import assert_raises

```
### 92.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 14
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import bootstrap_point632_score
from mlxtend.utils import assert_raises


```
### 93.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 109
```python
def test_scoring():
    from sklearn.metrics import f1_score

    lr = LogisticRegression(solver="liblinear", multi_class="ovr")
    scores = bootstrap_point632_score(
        lr, X[:100], y[:100], scoring_func=f1_score, random_seed=123
    )

```
### 94.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 121
```python
def test_scoring_proba():
    from sklearn.metrics import roc_auc_score

    lr = LogisticRegression(solver="liblinear", multi_class="ovr")

    # test predict_proba
    scores = bootstrap_point632_score(

```
### 95.
**path**: `.repositories/mlxtend/mlxtend/evaluate/holdout.py`
**line number**: 10
```python
import numpy as np
from sklearn.model_selection import train_test_split


class RandomHoldoutSplit(object):
    """Train/Validation set splitter for sklearn's GridSearchCV etc.


```
### 96.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 14
```python
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

```
### 97.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 15
```python
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB

```
### 98.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 16
```python
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

```
### 99.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 17
```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

```
### 100.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 18
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


```
### 101.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 19
```python
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingClassifier

```
### 102.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 20
```python
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingClassifier
from mlxtend.data import iris_data

```
### 103.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 21
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingClassifier
from mlxtend.data import iris_data
from mlxtend.externals.estimator_checks import NotFittedError

```
### 104.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_classifier.py`
**line number**: 22
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingClassifier
from mlxtend.data import iris_data
from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.utils import assert_raises

```
### 105.
**path**: `.repositories/mlxtend/mlxtend/classifier/ensemble_vote.py`
**line number**: 14
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from ..externals.name_estimators import _name_estimators


```
### 106.
**path**: `.repositories/mlxtend/mlxtend/classifier/ensemble_vote.py`
**line number**: 15
```python
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from ..externals.name_estimators import _name_estimators



```
### 107.
**path**: `.repositories/mlxtend/mlxtend/classifier/ensemble_vote.py`
**line number**: 16
```python
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from ..externals.name_estimators import _name_estimators


class EnsembleVoteClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

```
### 108.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/utilities.py`
**line number**: 4
```python
import numpy as np
from sklearn.model_selection import cross_val_score


def _merge_lists(nested_list, high_level_indices=None):
    """
    merge elements of lists (of a nested_list) into one single tuple with elements

```
### 109.
**path**: `.repositories/mlxtend/mlxtend/preprocessing/transactionencoder.py`
**line number**: 9
```python
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class TransactionEncoder(BaseEstimator, TransformerMixin):
    """Encoder class for transaction data in Python lists


```
### 110.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 12
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


```
### 111.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 13
```python
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector

```
### 112.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 14
```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector


```
### 113.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 15
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector



```
### 114.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_column_selector.py`
**line number**: 16
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector


def test_ColumnSelector():

```
### 115.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 12
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

```
### 116.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 13
```python
from sklearn import __version__ as sklearn_version
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB

```
### 117.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 14
```python
from sklearn import exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

```
### 118.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 15
```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


```
### 119.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 16
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import EnsembleVoteClassifier

```
### 120.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 17
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data

```
### 121.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 18
```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises

```
### 122.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_ensemble_vote_classifier.py`
**line number**: 19
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 123.
**path**: `.repositories/mlxtend/mlxtend/classifier/_base_classification.py`
**line number**: 3
```python
from scipy import sparse
from sklearn.base import ClassifierMixin

from ..externals.estimator_checks import check_is_fitted


class _BaseStackingClassifier(ClassifierMixin):

```
### 124.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 15
```python
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn import datasets, exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score

```
### 125.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 16
```python
from sklearn import __version__ as sklearn_version
from sklearn import datasets, exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (

```
### 126.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 17
```python
from sklearn import datasets, exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,

```
### 127.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 18
```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,

```
### 128.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 19
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,

```
### 129.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 20
```python
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,

```
### 130.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 21
```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)

```
### 131.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 27
```python
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingCVClassifier
from mlxtend.data import iris_data

```
### 132.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 28
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingCVClassifier
from mlxtend.data import iris_data
from mlxtend.externals.estimator_checks import NotFittedError

```
### 133.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_stacking_cv_classifier.py`
**line number**: 29
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingCVClassifier
from mlxtend.data import iris_data
from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.utils import assert_raises

```
### 134.
**path**: `.repositories/mlxtend/mlxtend/cluster/tests/test_kmeans.py`
**line number**: 8
```python
import numpy as np
from sklearn.base import clone

from mlxtend.cluster import Kmeans
from mlxtend.data import three_blobs_data
from mlxtend.utils import assert_raises


```
### 135.
**path**: `.repositories/mlxtend/mlxtend/evaluate/bootstrap_point632.py`
**line number**: 12
```python
import numpy as np
from sklearn.base import clone

from .bootstrap_outofbag import BootstrapOutOfBag


def _check_arrays(X, y=None):

```
### 136.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_cv_regression.py`
**line number**: 18
```python
from scipy import sparse
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted

```
### 137.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_cv_regression.py`
**line number**: 19
```python
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators

```
### 138.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_cv_regression.py`
**line number**: 20
```python
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition

```
### 139.
**path**: `.repositories/mlxtend/mlxtend/regressor/stacking_cv_regression.py`
**line number**: 21
```python
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition


```
### 140.
**path**: `.repositories/mlxtend/mlxtend/classifier/oner.py`
**line number**: 15
```python
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class OneRClassifier(BaseEstimator, ClassifierMixin):


```
### 141.
**path**: `.repositories/mlxtend/mlxtend/classifier/oner.py`
**line number**: 16
```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class OneRClassifier(BaseEstimator, ClassifierMixin):

    """OneR (One Rule) Classifier.

```
### 142.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py`
**line number**: 8
```python
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


```
### 143.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py`
**line number**: 9
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import boston_housing_data

```
### 144.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py`
**line number**: 10
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

```
### 145.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py`
**line number**: 11
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises

```
### 146.
**path**: `.repositories/mlxtend/mlxtend/feature_selection/tests/test_sequential_feature_selector_feature_groups.py`
**line number**: 12
```python
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


```
### 147.
**path**: `.repositories/mlxtend/mlxtend/classifier/stacking_classification.py`
**line number**: 15
```python
from scipy import sparse
from sklearn.base import TransformerMixin, clone

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from ._base_classification import _BaseStackingClassifier

```
### 148.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_oner.py`
**line number**: 8
```python
import numpy as np
from sklearn.model_selection import train_test_split

from mlxtend.classifier import OneRClassifier
from mlxtend.data import iris_data

X, y = iris_data()

```
### 149.
**path**: `.repositories/mlxtend/mlxtend/classifier/tests/test_logistic_regression.py`
**line number**: 10
```python
import numpy as np
from sklearn.base import clone

from mlxtend.classifier import LogisticRegression
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


```
### 150.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 11
```python
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

```
### 151.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 12
```python
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR


```
### 152.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 13
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

from mlxtend.evaluate import feature_importance_permutation

```
### 153.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 14
```python
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.utils import assert_raises

```
### 154.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 15
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.utils import assert_raises


```
### 155.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_feature_importance.py`
**line number**: 16
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.utils import assert_raises



```
### 156.
**path**: `.repositories/mlxtend/mlxtend/plotting/learning_curves.py`
**line number**: 78
```python
    if scoring != "misclassification error":
        from sklearn import metrics

        scoring_func = {
            "accuracy": metrics.accuracy_score,
            "average_precision": metrics.average_precision_score,
            "f1": metrics.f1_score,

```
### 157.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 12
```python
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

```
### 158.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 13
```python
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

```
### 159.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 14
```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


```
### 160.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 15
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError

```
### 161.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 16
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingRegressor

```
### 162.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 17
```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingRegressor
from mlxtend.utils import assert_raises

```
### 163.
**path**: `.repositories/mlxtend/mlxtend/regressor/tests/test_stacking_regression.py`
**line number**: 18
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingRegressor
from mlxtend.utils import assert_raises


```
### 164.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_pca_corr_graph.py`
**line number**: 3
```python
from packaging.version import Version
from sklearn import __version__ as sklearn_version

from mlxtend.data import iris_data
from mlxtend.plotting import plot_pca_correlation_graph

if Version(sklearn_version) < "0.22":

```
### 165.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_pca_corr_graph.py`
**line number**: 9
```python
if Version(sklearn_version) < "0.22":
    from sklearn.decomposition.pca import PCA
else:
    from sklearn.decomposition import PCA


def test_pass_pca_corr():

```
### 166.
**path**: `.repositories/mlxtend/mlxtend/plotting/tests/test_pca_corr_graph.py`
**line number**: 11
```python
else:
    from sklearn.decomposition import PCA


def test_pass_pca_corr():
    X, y = iris_data()
    plot_pca_correlation_graph(X, variables_names=["1", "2", "3", "4"])

```
### 167.
**path**: `.repositories/mlxtend/mlxtend/evaluate/time_series.py`
**line number**: 14
```python
from matplotlib.ticker import MaxNLocator
from sklearn.utils import indexable


class GroupTimeSeriesSplit:
    """Group time series cross-validator.


```
## tensorflow
### 1.
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
### 2.
**path**: `.repositories/mlxtend/mlxtend/evaluate/tests/test_bootstrap_point632.py`
**line number**: 163
```python
def test_keras_fitparams():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(32, activation=tf.nn.relu), tf.keras.layers.Dense(1)]
    )


```
