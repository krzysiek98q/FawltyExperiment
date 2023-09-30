# undeclared dependenciec
## AppKit
### 1.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 560
```python
        try:
            import AppKit
            import Foundation  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

```
## Foundation
### 1.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 561
```python
            import AppKit
            import Foundation  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()


```
## IPython
### 1.
**path**: `.repositories/pandas/pandas/io/formats/printing.py`
**line number**: 246
```python
        return
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        # still not in IPython
        return

```
### 2.
**path**: `.repositories/pandas/pandas/io/formats/printing.py`
**line number**: 259
```python
            # define tableschema formatter
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)

```
### 3.
**path**: `.repositories/pandas/pandas/tests/arrays/categorical/test_warnings.py`
**line number**: 13
```python
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; c = pd.Categorical([])"
        await ip.run_code(code)

        # GH 31324 newer jedi version raises Deprecation warning;

```
### 4.
**path**: `.repositories/pandas/pandas/conftest.py`
**line number**: 1850
```python
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()

```
### 5.
**path**: `.repositories/pandas/pandas/tests/frame/test_api.py`
**line number**: 295
```python
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        if frame_or_series is DataFrame:
            code = "from pandas import DataFrame; obj = DataFrame()"
        else:
            code = "from pandas import Series; obj = Series(dtype=object)"

```
### 6.
**path**: `.repositories/pandas/pandas/tests/resample/test_resampler_grouper.py`
**line number**: 31
```python
async def test_tab_complete_ipython6_warning(ip):
    from IPython.core.completer import provisionalcompleter

    code = dedent(
        """\
    import pandas._testing as tm
    s = tm.makeTimeSeries()

```
### 7.
**path**: `.repositories/pandas/pandas/tests/indexes/test_base.py`
**line number**: 1225
```python
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; idx = pd.Index([1, 2])"
        await ip.run_code(code)

        # GH 31324 newer jedi version raises Deprecation warning;

```
## PyQt4
### 1.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 147
```python
        except ImportError:
            from PyQt4.QtGui import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])


```
### 2.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 589
```python
                try:
                    import PyQt4  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.
                else:
                    return init_qt_clipboard()
            else:

```
## __main__
### 1.
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
## _csv
### 1.
**path**: `.repositories/pandas/pandas/tests/io/formats/test_to_csv.py`
**line number**: 6
```python

from _csv import Error
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
## boto3
### 1.
**path**: `.repositories/pandas/pandas/tests/io/conftest.py`
**line number**: 133
```python
def s3_resource(s3_base):
    import boto3

    s3 = boto3.resource("s3", endpoint_url=s3_base)
    return s3



```
## botocore
### 1.
**path**: `.repositories/pandas/pandas/tests/io/test_s3.py`
**line number**: 12
```python
    pytest.importorskip("botocore", minversion="1.10.47")
    from botocore.response import StreamingBody

    data = [b"foo,bar,baz\n1,2,3\n4,5,6\n", b"just,the,header\n"]
    for el in data:
        body = StreamingBody(BytesIO(el), content_length=len(el))
        read_csv(body)

```
### 2.
**path**: `.repositories/pandas/pandas/io/common.py`
**line number**: 417
```python
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            err_types_to_retry_with_anon = [

```
### 3.
**path**: `.repositories/pandas/pandas/tests/io/parser/test_network.py`
**line number**: 244
```python
        # Attempting to write to an invalid S3 path should raise
        import botocore

        # GH 34087
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        # Catch a ClientError since AWS Service Errors are defined dynamically
        error = (FileNotFoundError, botocore.exceptions.ClientError)

```
### 4.
**path**: `.repositories/pandas/pandas/tests/io/parser/test_network.py`
**line number**: 261
```python
        pytest.importorskip("pyarrow")
        import botocore

        # GH 34087
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        # Catch a ClientError since AWS Service Errors are defined dynamically
        error = (FileNotFoundError, botocore.exceptions.ClientError)

```
## cycler
### 1.
**path**: `.repositories/pandas/pandas/tests/plotting/frame/test_frame_color.py`
**line number**: 596
```python
    def test_default_color_cycle(self):
        import cycler

        colors = list("rgbk")
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

```
## dask
### 1.
**path**: `.repositories/pandas/pandas/tests/test_downstream.py`
**line number**: 292
```python
    elif name == "dask":
        import dask.array

        data = dask.array.array(arr)
    elif name == "xarray":
        import xarray as xr


```
## google
### 1.
**path**: `.repositories/pandas/pandas/io/gbq.py`
**line number**: 12
```python
if TYPE_CHECKING:
    import google.auth

    from pandas import DataFrame


def _try_import():

```
## markupsafe
### 1.
**path**: `.repositories/pandas/pandas/io/formats/style_render.py`
**line number**: 49
```python
jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency

BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = tuple[str, Union[str, float]]
CSSList = list[CSSPair]

```
## python_calamine
### 1.
**path**: `.repositories/pandas/pandas/tests/io/excel/test_readers.py`
**line number**: 880
```python
        elif engine == "calamine":
            from python_calamine import CalamineError

            error = CalamineError
            msg = "Cannot detect file format"
        else:
            error = BadZipFile

```
### 2.
**path**: `.repositories/pandas/pandas/tests/io/excel/test_readers.py`
**line number**: 1741
```python
        elif engine == "calamine":
            from python_calamine import CalamineError

            errors = (CalamineError,)

        with tm.ensure_clean(f"corrupt{read_ext}") as file:
            Path(file).write_text("corrupt", encoding="utf-8")

```
### 3.
**path**: `.repositories/pandas/pandas/io/excel/_calamine.py`
**line number**: 26
```python
if TYPE_CHECKING:
    from python_calamine import (
        CalamineSheet,
        CalamineWorkbook,
    )

    from pandas._typing import (

```
### 4.
**path**: `.repositories/pandas/pandas/io/excel/_calamine.py`
**line number**: 68
```python
    def _workbook_class(self) -> type[CalamineWorkbook]:
        from python_calamine import CalamineWorkbook

        return CalamineWorkbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any

```
### 5.
**path**: `.repositories/pandas/pandas/io/excel/_calamine.py`
**line number**: 75
```python
    ) -> CalamineWorkbook:
        from python_calamine import load_workbook

        return load_workbook(
            filepath_or_buffer, **engine_kwargs  # type: ignore[arg-type]
        )


```
### 6.
**path**: `.repositories/pandas/pandas/io/excel/_calamine.py`
**line number**: 83
```python
    def sheet_names(self) -> list[str]:
        from python_calamine import SheetTypeEnum

        return [
            sheet.name
            for sheet in self.book.sheets_metadata
            if sheet.typ == SheetTypeEnum.WorkSheet

```
## sklearn
### 1.
**path**: `.repositories/pandas/pandas/tests/test_downstream.py`
**line number**: 147
```python
    pytest.importorskip("sklearn")
    from sklearn import (
        datasets,
        svm,
    )

    digits = datasets.load_digits()

```
## traitlets
### 1.
**path**: `.repositories/pandas/pandas/io/formats/printing.py`
**line number**: 260
```python
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)


```
### 2.
**path**: `.repositories/pandas/pandas/conftest.py`
**line number**: 1853
```python
    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()
    c.HistoryManager.hist_file = ":memory:"

    return InteractiveShell(config=c)

```
## typing_extensions
### 1.
**path**: `.repositories/pandas/pandas/_typing.py`
**line number**: 94
```python
    else:
        from typing_extensions import TypeGuard  # pyright: ignore[reportUnusedImport]

    if sys.version_info >= (3, 11):
        from typing import Self  # pyright: ignore[reportUnusedImport]
    else:
        from typing_extensions import Self  # pyright: ignore[reportUnusedImport]

```
### 2.
**path**: `.repositories/pandas/pandas/_typing.py`
**line number**: 99
```python
    else:
        from typing_extensions import Self  # pyright: ignore[reportUnusedImport]
else:
    npt: Any = None
    Self: Any = None
    TypeGuard: Any = None


```
