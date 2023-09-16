# undeclared dependenciec
## IPython
### 1.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 58
```python
    try:
        from IPython.display import display  # , clear_output
    except ImportError:
        pass

__author__ = {"github.com/": ["lrq3000", "casperdcl", "alexanderkuk"]}
__all__ = ['tqdm_notebook', 'tnrange', 'tqdm', 'trange']

```
### 2.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 38
```python
        if IPY == 32:
            from IPython.html.widgets import HTML
            from IPython.html.widgets import FloatProgress as IProgress
            from IPython.html.widgets import HBox
            IPY = 3
        else:
            from ipywidgets import HTML

```
### 3.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 39
```python
            from IPython.html.widgets import HTML
            from IPython.html.widgets import FloatProgress as IProgress
            from IPython.html.widgets import HBox
            IPY = 3
        else:
            from ipywidgets import HTML
            from ipywidgets import FloatProgress as IProgress

```
### 4.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 40
```python
            from IPython.html.widgets import FloatProgress as IProgress
            from IPython.html.widgets import HBox
            IPY = 3
        else:
            from ipywidgets import HTML
            from ipywidgets import FloatProgress as IProgress
            from ipywidgets import HBox

```
### 5.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 48
```python
        try:  # IPython 2.x
            from IPython.html.widgets import HTML
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import FloatProgressWidget as IProgress
            IPY = 2
        except ImportError:
            IPY = 0

```
### 6.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 49
```python
            from IPython.html.widgets import HTML
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import FloatProgressWidget as IProgress
            IPY = 2
        except ImportError:
            IPY = 0
            IProgress = None

```
### 7.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 50
```python
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import FloatProgressWidget as IProgress
            IPY = 2
        except ImportError:
            IPY = 0
            IProgress = None
            HBox = object

```
### 8.
**path**: `.repositories/tqdm/tqdm/notebook.py`
**line number**: 32
```python
            try:
                import IPython.html.widgets as ipywidgets  # NOQA: F401
            except ImportError:
                pass

    try:  # IPython 4.x / 3.x
        if IPY == 32:

```
## disco
### 1.
**path**: `.repositories/tqdm/tqdm/contrib/discord.py`
**line number**: 15
```python
try:
    from disco.client import Client, ClientConfig
except ImportError:
    raise ImportError("Please `pip install disco-py`")

from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker

```
## matplotlib
### 1.
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
### 2.
**path**: `.repositories/tqdm/tqdm/gui.py`
**line number**: 31
```python
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        colour = kwargs.pop('colour', 'g')
        super(tqdm_gui, self).__init__(*args, **kwargs)


```
## numpy
### 1.
**path**: `.repositories/tqdm/tqdm/contrib/__init__.py`
**line number**: 59
```python
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(iterable, np.ndarray):
            return tqdm_class(np.ndenumerate(iterable), total=total or iterable.size,

```
## pandas
### 1.
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
### 2.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 808
```python
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
        try:
            with catch_warnings():
                simplefilter("ignore", category=FutureWarning)
                from pandas import Panel
        except ImportError:  # pandas>=1.2.0

```
### 3.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 817
```python
        try:  # pandas>=1.0.0
            from pandas.core.window.rolling import _Rolling_and_Expanding
        except ImportError:
            try:  # pandas>=0.18.0
                from pandas.core.window import _Rolling_and_Expanding
            except ImportError:  # pandas>=1.2.0
                try:  # pandas>=1.2.0

```
### 4.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 829
```python
        try:  # pandas>=0.25.0
            from pandas.core.groupby.generic import SeriesGroupBy  # , NDFrameGroupBy
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError:  # pragma: no cover
            try:  # pandas>=0.23.0
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:

```
### 5.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 830
```python
            from pandas.core.groupby.generic import SeriesGroupBy  # , NDFrameGroupBy
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError:  # pragma: no cover
            try:  # pandas>=0.23.0
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

```
### 6.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 837
```python
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:  # pragma: no cover
            from pandas.core.groupby import GroupBy

        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import PanelGroupBy

```
### 7.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 842
```python
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import PanelGroupBy
        except ImportError:
            try:
                from pandas.core.groupby import PanelGroupBy
            except ImportError:  # pandas>=0.25.0
                PanelGroupBy = None

```
### 8.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 812
```python
                simplefilter("ignore", category=FutureWarning)
                from pandas import Panel
        except ImportError:  # pandas>=1.2.0
            Panel = None
        Rolling, Expanding = None, None
        try:  # pandas>=1.0.0
            from pandas.core.window.rolling import _Rolling_and_Expanding

```
### 9.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 839
```python
        except ImportError:  # pragma: no cover
            from pandas.core.groupby import GroupBy

        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import PanelGroupBy
        except ImportError:
            try:

```
### 10.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 820
```python
            try:  # pandas>=0.18.0
                from pandas.core.window import _Rolling_and_Expanding
            except ImportError:  # pandas>=1.2.0
                try:  # pandas>=1.2.0
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling
                    _Rolling_and_Expanding = Rolling, Expanding

```
### 11.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 833
```python
            try:  # pandas>=0.23.0
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:  # pragma: no cover

```
### 12.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 845
```python
            try:
                from pandas.core.groupby import PanelGroupBy
            except ImportError:  # pandas>=0.25.0
                PanelGroupBy = None

        tqdm_kwargs = tqdm_kwargs.copy()
        deprecated_t = [tqdm_kwargs.pop('deprecated_t', None)]

```
### 13.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 900
```python
                try:  # pandas>=1.3.0
                    from pandas.core.common import is_builtin_func
                except ImportError:
                    is_builtin_func = df._is_builtin_func
                try:
                    func = is_builtin_func(func)
                except TypeError:

```
### 14.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 835
```python
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:  # pragma: no cover
            from pandas.core.groupby import GroupBy


```
### 15.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 823
```python
                try:  # pandas>=1.2.0
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling
                    _Rolling_and_Expanding = Rolling, Expanding
                except ImportError:  # pragma: no cover
                    _Rolling_and_Expanding = None
        try:  # pandas>=0.25.0

```
### 16.
**path**: `.repositories/tqdm/tqdm/std.py`
**line number**: 824
```python
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling
                    _Rolling_and_Expanding = Rolling, Expanding
                except ImportError:  # pragma: no cover
                    _Rolling_and_Expanding = None
        try:  # pandas>=0.25.0
            from pandas.core.groupby.generic import SeriesGroupBy  # , NDFrameGroupBy

```
## setuptools_scm
### 1.
**path**: `.repositories/tqdm/tqdm/version.py`
**line number**: 6
```python
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"

```
## tensorflow
### 1.
**path**: `.repositories/tqdm/tqdm/keras.py`
**line number**: 10
```python
    try:
        from tensorflow import keras
    except ImportError:
        raise e
__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['TqdmCallback']


```
