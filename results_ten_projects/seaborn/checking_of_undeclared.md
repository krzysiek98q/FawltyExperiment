# undeclared dependenciec
## PIL
### 1.
**path**: `.repositories/seaborn/seaborn/_core/plot.py`
**line number**: 24
```python
import numpy as np
from PIL import Image

from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move

```
### 2.
**path**: `.repositories/seaborn/seaborn/palettes.py`
**line number**: 99
```python
        import io
        from PIL import Image
        import numpy as np
        IMAGE_SIZE = (400, 50)
        X = np.tile(np.linspace(0, 1, IMAGE_SIZE[0]), (IMAGE_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()

```
## com
### 1.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 208
```python
    import array
    from com.sun import jna
    from com.sun.jna.platform import win32

    buf_size = win32.WinDef.MAX_PATH * 2
    buf = array.zeros('c', buf_size)
    shell = win32.Shell32.INSTANCE

```
### 2.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 209
```python
    from com.sun import jna
    from com.sun.jna.platform import win32

    buf_size = win32.WinDef.MAX_PATH * 2
    buf = array.zeros('c', buf_size)
    shell = win32.Shell32.INSTANCE
    shell.SHGetFolderPath(None, getattr(win32.ShlObj, csidl_name), None, win32.ShlObj.SHGFP_TYPE_CURRENT, buf)

```
### 3.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 242
```python
            try:
                import com.sun.jna
                _get_win_folder = _get_win_folder_with_jna
            except ImportError:
                _get_win_folder = _get_win_folder_from_registry

```
## cycler
### 1.
**path**: `.repositories/seaborn/seaborn/_core/plot.py`
**line number**: 16
```python

from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist

```
### 2.
**path**: `.repositories/seaborn/seaborn/rcmod.py`
**line number**: 4
```python
import matplotlib as mpl
from cycler import cycler
from . import palettes


__all__ = ["set_theme", "set", "reset_defaults", "reset_orig",
           "axes_style", "set_style", "plotting_context", "set_context",

```
## fastcluster
### 1.
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
## ipywidgets
### 1.
**path**: `.repositories/seaborn/seaborn/widgets.py`
**line number**: 6
```python
try:
    from ipywidgets import interact, FloatSlider, IntSlider
except ImportError:
    def interact(f):
        msg = "Interactive palettes require `ipywidgets`, which is not installed."
        raise ImportError(msg)


```
## sphinx
### 1.
**path**: `.repositories/seaborn/seaborn/external/docscrape.py`
**line number**: 645
```python
        if 'sphinx' in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        self.show_inherited_members = config.get(
                    'show_inherited_class_members', True)

```
## win32api
### 1.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 171
```python
            try:
                import win32api
                dir = win32api.GetShortPathName(dir)
            except ImportError:
                pass
    except UnicodeError:
        pass

```
## win32com
### 1.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 154
```python
def _get_win_folder_with_pywin32(csidl_name):
    from win32com.shell import shellcon, shell
    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    # Try to make this a unicode path because SHGetFolderPath does
    # not return unicode strings when there is unicode data in the
    # path.
    try:

```
### 2.
**path**: `.repositories/seaborn/seaborn/external/appdirs.py`
**line number**: 234
```python
    try:
        import win32com.shell
        _get_win_folder = _get_win_folder_with_pywin32
    except ImportError:
        try:
            from ctypes import windll
            _get_win_folder = _get_win_folder_with_ctypes

```
