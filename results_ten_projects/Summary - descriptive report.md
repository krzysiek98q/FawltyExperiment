# Summary - descriptive report

This report is a brief overview of the achieved results of FawltyDeps. The description excludes problems related to package name mapping, since FawltyDeps in theory should handle this using the `--install-deps` option. Unfortunately, with this experiment, this option makes it difficult to obtain consistent results, so it was not used.

In this file you will find, a brief description of what is in the files `checking_of_undeclared.md` and `summary.toml`. If you want to see more details, please review e files.

## Kivy

FawltyDeps found libraries that are used but not specified in the requirements, for example `numpy`.

**path**: `.repositories/Kivy/kivy/core/camera/camera_android.py` 

**line number**: 187

```python
        """
        import numpy as np
        from cv2 import cvtColor

        w, h = self._resolution
        arr = np.fromstring(buf, 'uint8').reshape((h + h / 2, w))
        arr = cvtColor(arr, 93)  # NV21 -> BGR
```

The first problem we may notice is that packages such as `kivy-deps.angle`, `kivy-deps.glew`, `kivy-deps.sdl2` are in the declared dependencies, but are treated as unused dependencies. These packages are probably part of Kiva helping the main library work.

The biggest questionable result of FwaltyDeps is the `undeclared_deps` list. A common problem is that code developers put imports in the `try...except` (and `if...else`) block. FawltyDeps detects these imports as undeclared dependencies. A good solution would be to not treat these as required dependencies. 

- **Image**
  
  - path: `.repositories/Kivy/kivy/core/image/img_pil.py` 
  
  - line number: 8

```python
try:
    import Image as PILImage
except ImportError:
    # for python3
    from PIL import Image as PILImage

from kivy.logger import Logger
```

- **android**
  
  - path: `.repositories/Kivy/kivy/core/window/window_pygame.py` 
  
  - line number: 31

```python
    if platform == 'android':
        import android
except ImportError:
    pass

# late binding
glReadPixels = GL_RGBA = GL_UNSIGNED_BYTE = None
```

No warnings appeared during the operation of FawltyDeps.

## mlxtend

In this package, the unused dependencies are `mkdocs` and `nbconvert`. No significant undeclared dependencies were found, however FawltyDeps detected tensorflow as an undeclared dependency, but the example below shows that it was used in the test function.

**path**:  `.repositories/mlxtend/mlxtend/evaluate/tests/test_bias_variance_decomp.py` **line number**: 118

```python
def test_keras():
    import tensorflow as tf

    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True
    )
```

In addition, the `nose` library, also should not be treated absolutely as a dependency as the following code suggests.

**path**: `.repositories/mlxtend/mlxtend/text/tests/test_generalize_names.py` 

**line number**: 4

```python
if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest

    raise SkipTest

from mlxtend.text import generalize_names
```

No warnings appeared during the operation of FawltyDeps.

## nltk

No dependencies found that are unused. This package found `networkx` and `yaml` as undeclared dependencies. In addition, there was again a problem with FawltyDeps treating imports in the `try...except` block as undeclared dependencies.

**path**: `.repositories/nltk/nltk/parse/bllip.py`

**line number**: 87

```python
try:
    from bllipparser import RerankingParser
    from bllipparser.RerankingParser import get_unified_model_parameters

    def _ensure_bllip_import_or_error():
        pass
```

No unused dependencies found.

## pandas

A great deal of undeclared dependencies were found in pandas, the occurrence of which is unlikely to be an error due to the fact that python is a dynamic language.

The presence of `__main__` in undeclared dependencies may be questionable.

**path**: `.repositories/pandas/pandas/io/formats/console.py` 

**line number**: 67

```python
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")

    try:
```

Besides, FawltyDeps found `boto3` or `google` packages in undeclared dependencies, which should not appear in this list. 

**path**: `.repositories/pandas/pandas/io/gbq.py` 

**line number**: 12

```python
if TYPE_CHECKING:
    import google.auth

    from pandas import DataFrame


def _try_import():
```

## pingouin

In the case of this package, no requirements were found in the api pypi, so all dependencies found are undeclared. Again, however, the dependency in the `try...except` block is treated as undeclared. 

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

## scikit-learn

Not a lot of unused dependencies were found. Among those found were `plotly` and `seaboern`. The presence of `sphinx*` in the undeclared dependencies may be questionable.
There are very few undeclared dependencies in this extensive library. Found among others `cupy` or `setuptools`. In this experiment, no dependencies were found in `try...except` blocks, most likely such practices do not exist among the developers of this package.

## scipy

A lot of unused dependencies were found in this library. Among the undeclared dependencies again is `Cython` and `cups`. Among the undeclared dependencies again are those in the `try...except` block.

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

There is also a `sphinx*` package in `scipy` that could be ignored.

## seaborn

Relatively few undeclared dependencies were found in this library. It can be noted that  `fastcluster` and `pillow` packages were not included in the library requirements.

**path**: `.repositories/seaborn/seaborn/matrix.py` 

**line number**: 535

```python
    def _calculate_linkage_fastcluster(self):
        import fastcluster
        # Fastcluster has a memory-saving vectorized version, but only
        # with certain linkage methods, and mostly with euclidean metric
        # vector_methods = ('single', 'centroid', 'median', 'ward')
        euclidean_methods = ('centroid', 'median', 'ward')
        euclidean = self.metric == 'euclidean' and self.method in \
```

## statsmodels

In this library, undeclared dependencies occur mainly in the tests/examples files. Besides, again, the most common undeclared dependencies are found in `try...except` blocks.

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

## tqdm

The only undeclared dependencies found that are not in the `try..except` block are `matplotlib` and `pandas`. 

**path**: `.repositories/tqdm/tqdm/gui.py` 

**line number**: 30

```python
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        colour = kwargs.pop('colour', 'g')
        super(tqdm_gui, self).__init__(*args, **kwargs)
```

**path**: `.repositories/tqdm/tqdm/std.py` 

**line number**: 807

```python
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
        try:
            with catch_warnings():
                simplefilter("ignore", category=FutureWarning)
                from pandas import Panel
```

# Summary

No problems were found in the operation of FawltyDeps. This package was able to find dependencies that were used but not declared. 

One can only note the frequent finding of imports in `try...except` or `if...else` blocks. You might consider omitting such dependencies in future versions of FawltyDeps, since these blocks most often contain support for specific libraries that may not be considered required.
