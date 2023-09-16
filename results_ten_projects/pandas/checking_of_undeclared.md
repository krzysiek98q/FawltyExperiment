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
### 2.
**path**: `.repositories/pandas/pandas/tests/indexes/test_base.py`
**line number**: 1225
```python
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; idx = pd.Index([1, 2])"
        await ip.run_code(code)

        # GH 31324 newer jedi version raises Deprecation warning;

```
### 3.
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
### 4.
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
**path**: `.repositories/pandas/pandas/conftest.py`
**line number**: 1848
```python
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()

```
### 7.
**path**: `.repositories/pandas/pandas/tests/arrays/categorical/test_warnings.py`
**line number**: 13
```python
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; c = pd.Categorical([])"
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
## PyQt5
<span style="color:red">**PyQt5 is in the dependencies declared**
</span>.### 1.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 145
```python
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication

    app = QApplication.instance()
    if app is None:

```
### 2.
**path**: `.repositories/pandas/pandas/io/clipboard/__init__.py`
**line number**: 586
```python
            try:
                import PyQt5  # check if PyQt5 is installed
            except ImportError:
                try:
                    import PyQt4  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.

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
## bs4
### 1.
**path**: `.repositories/pandas/pandas/io/html.py`
**line number**: 595
```python
        super().__init__(*args, **kwargs)
        from bs4 import SoupStrainer

        self._strainer = SoupStrainer("table")

    def _parse_tables(self, document, match, attrs):
        element_name = self._strainer.name

```
### 2.
**path**: `.repositories/pandas/pandas/io/html.py`
**line number**: 656
```python
    def _build_doc(self):
        from bs4 import BeautifulSoup

        bdoc = self._setup_build_doc()
        if isinstance(bdoc, bytes) and self.encoding is not None:
            udoc = bdoc.decode(self.encoding)
            from_encoding = None

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
## dateutil
### 1.
**path**: `.repositories/pandas/pandas/tests/indexing/test_loc.py`
**line number**: 11
```python

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas.errors import IndexingError
import pandas.util._test_decorators as td

```
### 2.
**path**: `.repositories/pandas/pandas/core/indexes/datetimes.py`
**line number**: 746
```python
        if isinstance(time, str):
            from dateutil.parser import parse

            time = parse(time).time()

        if time.tzinfo:
            if self.tz is None:

```
### 3.
**path**: `.repositories/pandas/pandas/tests/tslibs/test_timezones.py`
**line number**: 7
```python

import dateutil.tz
import pytest
import pytz

from pandas._libs.tslibs import (
    conversion,

```
### 4.
**path**: `.repositories/pandas/pandas/tests/reshape/concat/test_datetimes.py`
**line number**: 4
```python

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
### 5.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_datetime.py`
**line number**: 3
```python

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
### 6.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/methods/test_to_period.py`
**line number**: 1
```python
import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz

from pandas._libs.tslibs.ccalendar import MONTHS

```
### 7.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/methods/test_to_period.py`
**line number**: 2
```python
import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz

from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd

```
### 8.
**path**: `.repositories/pandas/pandas/tests/io/formats/test_format.py`
**line number**: 20
```python

import dateutil
import numpy as np
import pytest
import pytz

from pandas._config import config

```
### 9.
**path**: `.repositories/pandas/pandas/tests/tslibs/test_parsing.py`
**line number**: 7
```python

from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas._libs.tslibs import (

```
### 10.
**path**: `.repositories/pandas/pandas/tests/tslibs/test_parsing.py`
**line number**: 8
```python
from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas._libs.tslibs import (
    parsing,

```
### 11.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_unary_ops.py`
**line number**: 3
```python

from dateutil.tz import gettz
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np

```
### 12.
**path**: `.repositories/pandas/pandas/tests/frame/methods/test_dropna.py`
**line number**: 3
```python

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
### 13.
**path**: `.repositories/pandas/pandas/tests/io/json/test_ujson.py`
**line number**: 10
```python

import dateutil
import numpy as np
import pytest
import pytz

import pandas._libs.json as ujson

```
### 14.
**path**: `.repositories/pandas/pandas/tests/series/test_constructors.py`
**line number**: 8
```python

from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest

from pandas._libs import (

```
### 15.
**path**: `.repositories/pandas/pandas/tests/tseries/offsets/test_fiscal.py`
**line number**: 6
```python

from dateutil.relativedelta import relativedelta
import pytest

from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
    WeekDay,

```
### 16.
**path**: `.repositories/pandas/pandas/tests/io/sas/test_sas7bdat.py`
**line number**: 7
```python

import dateutil.parser
import numpy as np
import pytest

from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td

```
### 17.
**path**: `.repositories/pandas/pandas/tests/tseries/offsets/test_common.py`
**line number**: 3
```python

from dateutil.tz.tz import tzlocal
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,

```
### 18.
**path**: `.repositories/pandas/pandas/tests/frame/test_reductions.py`
**line number**: 5
```python

from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas.compat import (
    IS64,

```
### 19.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_timezones.py`
**line number**: 12
```python

import dateutil
from dateutil.tz import (
    gettz,
    tzoffset,
)
import pytest

```
### 20.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_timezones.py`
**line number**: 13
```python
import dateutil
from dateutil.tz import (
    gettz,
    tzoffset,
)
import pytest
import pytz

```
### 21.
**path**: `.repositories/pandas/pandas/tests/reshape/concat/test_append.py`
**line number**: 4
```python

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
### 22.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_constructors.py`
**line number**: 11
```python

import dateutil
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (

```
### 23.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_timestamp.py`
**line number**: 13
```python

from dateutil.tz import (
    tzlocal,
    tzutc,
)
from hypothesis import (
    given,

```
### 24.
**path**: `.repositories/pandas/pandas/tests/resample/test_period_index.py`
**line number**: 3
```python

import dateutil
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.ccalendar import (

```
### 25.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_formats.py`
**line number**: 3
```python

import dateutil.tz
import numpy as np
import pytest
import pytz

import pandas as pd

```
### 26.
**path**: `.repositories/pandas/pandas/tests/io/parser/test_converters.py`
**line number**: 7
```python

from dateutil.parser import parse
import numpy as np
import pytest

import pandas as pd
from pandas import (

```
### 27.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_ops.py`
**line number**: 3
```python

from dateutil.tz import tzlocal
import pytest

from pandas.compat import IS64

from pandas import (

```
### 28.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/methods/test_astype.py`
**line number**: 3
```python

import dateutil
import numpy as np
import pytest
import pytz

import pandas as pd

```
### 29.
**path**: `.repositories/pandas/pandas/tests/series/indexing/test_datetime.py`
**line number**: 10
```python

from dateutil.tz import (
    gettz,
    tzutc,
)
import numpy as np
import pytest

```
### 30.
**path**: `.repositories/pandas/pandas/conftest.py`
**line number**: 39
```python

from dateutil.tz import (
    tzlocal,
    tzutc,
)
import hypothesis
from hypothesis import strategies as st

```
### 31.
**path**: `.repositories/pandas/pandas/tests/tslibs/test_array_to_datetime.py`
**line number**: 8
```python

from dateutil.tz.tz import tzoffset
import numpy as np
import pytest

from pandas._libs import (
    iNaT,

```
### 32.
**path**: `.repositories/pandas/pandas/tests/io/parser/test_parse_dates.py`
**line number**: 14
```python

from dateutil.parser import parse as du_parse
from hypothesis import given
import numpy as np
import pytest
import pytz


```
### 33.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_constructors.py`
**line number**: 10
```python

import dateutil.tz
from dateutil.tz import tzutc
import numpy as np
import pytest
import pytz


```
### 34.
**path**: `.repositories/pandas/pandas/tests/scalar/timestamp/test_constructors.py`
**line number**: 11
```python
import dateutil.tz
from dateutil.tz import tzutc
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

```
### 35.
**path**: `.repositories/pandas/pandas/tseries/holiday.py`
**line number**: 9
```python

from dateutil.relativedelta import (
    FR,
    MO,
    SA,
    SU,
    TH,

```
### 36.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_timezones.py`
**line number**: 13
```python

import dateutil
from dateutil.tz import (
    gettz,
    tzlocal,
)
import numpy as np

```
### 37.
**path**: `.repositories/pandas/pandas/tests/indexes/datetimes/test_timezones.py`
**line number**: 14
```python
import dateutil
from dateutil.tz import (
    gettz,
    tzlocal,
)
import numpy as np
import pytest

```
### 38.
**path**: `.repositories/pandas/pandas/tests/tools/test_to_datetime.py`
**line number**: 14
```python

from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz


```
### 39.
**path**: `.repositories/pandas/pandas/tests/tools/test_to_datetime.py`
**line number**: 15
```python
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz

from pandas._libs import tslib

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
## mpl_toolkits
### 1.
**path**: `.repositories/pandas/pandas/tests/plotting/frame/test_frame.py`
**line number**: 2224
```python
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=cax)

```
### 2.
**path**: `.repositories/pandas/pandas/tests/plotting/frame/test_frame.py`
**line number**: 2233
```python
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        iax = inset_axes(ax, width="30%", height=1.0, loc=3)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=iax)


```
## odf
### 1.
**path**: `.repositories/pandas/pandas/tests/io/excel/test_odswriter.py`
**line number**: 76
```python
    # http://docs.oasis-open.org/office/v1.2/os/OpenDocument-v1.2-os-part1.html#refTable13
    from odf.namespaces import OFFICENS
    from odf.table import (
        TableCell,
        TableRow,
    )


```
### 2.
**path**: `.repositories/pandas/pandas/tests/io/excel/test_odswriter.py`
**line number**: 77
```python
    from odf.namespaces import OFFICENS
    from odf.table import (
        TableCell,
        TableRow,
    )

    table_cell_name = TableCell().qname

```
### 3.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 25
```python
if TYPE_CHECKING:
    from odf.opendocument import OpenDocument

    from pandas._libs.tslibs.nattype import NaTType


@doc(storage_options=_shared_docs["storage_options"])

```
### 4.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 58
```python
    def _workbook_class(self) -> type[OpenDocument]:
        from odf.opendocument import OpenDocument

        return OpenDocument

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs

```
### 5.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 65
```python
    ) -> OpenDocument:
        from odf.opendocument import load

        return load(filepath_or_buffer, **engine_kwargs)

    @property
    def empty_value(self) -> str:

```
### 6.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 77
```python
        """Return a list of sheet names present in the document"""
        from odf.table import Table

        tables = self.book.getElementsByType(Table)
        return [t.getAttribute("name") for t in tables]

    def get_sheet_by_index(self, index: int):

```
### 7.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 83
```python
    def get_sheet_by_index(self, index: int):
        from odf.table import Table

        self.raise_if_bad_sheet_by_index(index)
        tables = self.book.getElementsByType(Table)
        return tables[index]


```
### 8.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 90
```python
    def get_sheet_by_name(self, name: str):
        from odf.table import Table

        self.raise_if_bad_sheet_by_name(name)
        tables = self.book.getElementsByType(Table)

        for table in tables:

```
### 9.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 108
```python
        """
        from odf.table import (
            CoveredTableCell,
            TableCell,
            TableRow,
        )


```
### 10.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 176
```python
        """
        from odf.namespaces import TABLENS

        return int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))

    def _get_column_repeat(self, cell) -> int:
        from odf.namespaces import TABLENS

```
### 11.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 181
```python
    def _get_column_repeat(self, cell) -> int:
        from odf.namespaces import TABLENS

        return int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))

    def _is_empty_row(self, row) -> bool:
        """

```
### 12.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 196
```python
    def _get_cell_value(self, cell) -> Scalar | NaTType:
        from odf.namespaces import OFFICENS

        if str(cell) == "#N/A":
            return np.nan

        cell_type = cell.attributes.get((OFFICENS, "value-type"))

```
### 13.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 239
```python
        """
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.text import S

        text_s = S().qname


```
### 14.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 240
```python
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.text import S

        text_s = S().qname

        value = []

```
### 15.
**path**: `.repositories/pandas/pandas/io/excel/_odfreader.py`
**line number**: 241
```python
        from odf.namespaces import TEXTNS
        from odf.text import S

        text_s = S().qname

        value = []


```
### 16.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 47
```python
    ) -> None:
        from odf.opendocument import OpenDocumentSpreadsheet

        if mode == "a":
            raise ValueError("Append mode is not supported with odf!")

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

```
### 17.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 77
```python
        """Mapping of sheet names to sheet objects."""
        from odf.table import Table

        result = {
            sheet.getAttribute("name"): sheet
            for sheet in self.book.getElementsByType(Table)
        }

```
### 18.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 104
```python
        """
        from odf.table import (
            Table,
            TableCell,
            TableRow,
        )
        from odf.text import P

```
### 19.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 109
```python
        )
        from odf.text import P

        sheet_name = self._get_sheet_name(sheet_name)
        assert sheet_name is not None

        if sheet_name in self.sheets:

```
### 20.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 187
```python
        """
        from odf.table import TableCell

        attributes = self._make_table_cell_attributes(cell)
        val, fmt = self._value_with_fmt(cell.val)
        pvalue = value = val
        if isinstance(val, bool):

```
### 21.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 261
```python
        """
        from odf.style import (
            ParagraphProperties,
            Style,
            TableCellProperties,
            TextProperties,
        )

```
### 22.
**path**: `.repositories/pandas/pandas/io/excel/_odswriter.py`
**line number**: 313
```python
        """
        from odf.config import (
            ConfigItem,
            ConfigItemMapEntry,
            ConfigItemMapIndexed,
            ConfigItemMapNamed,
            ConfigItemSet,

```
## py
### 1.
**path**: `.repositories/pandas/pandas/tests/io/excel/test_readers.py`
**line number**: 957
```python
        # GH12655
        from py.path import local as LocalPath

        str_path = os.path.join("test1" + read_ext)
        expected = pd.read_excel(str_path, sheet_name="Sheet1", index_col=0)

        path_obj = LocalPath().join("test1" + read_ext)

```
### 2.
**path**: `.repositories/pandas/pandas/tests/io/sas/test_sas7bdat.py`
**line number**: 86
```python
    def test_path_localpath(self, dirpath, data_test_ix):
        from py.path import local as LocalPath

        df0, test_ix = data_test_ix
        for k in test_ix:
            fname = LocalPath(os.path.join(dirpath, f"test{k}.sas7bdat"))
            df = pd.read_sas(fname, encoding="utf-8")

```
### 3.
**path**: `.repositories/pandas/pandas/tests/io/pytables/test_read.py`
**line number**: 337
```python
    # GH11773
    from py.path import local as LocalPath

    expected = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),

```
### 4.
**path**: `.repositories/pandas/pandas/tests/io/test_common.py`
**line number**: 43
```python
try:
    from py.path import local as LocalPath

    path_types.append(LocalPath)
except ImportError:
    pass


```
## pylab
### 1.
**path**: `.repositories/pandas/pandas/tests/plotting/test_hist_method.py`
**line number**: 137
```python
    def test_plot_fails_when_ax_differs_from_figure(self, ts):
        from pylab import figure

        fig1 = figure()
        fig2 = figure()
        ax1 = fig1.add_subplot(111)
        msg = "passed axis not bound to passed figure"

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
**line number**: 1851
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
**line number**: 91
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
**line number**: 96
```python
    else:
        from typing_extensions import Self  # pyright: ignore[reportUnusedImport]
else:
    npt: Any = None
    Self: Any = None
    TypeGuard: Any = None


```
