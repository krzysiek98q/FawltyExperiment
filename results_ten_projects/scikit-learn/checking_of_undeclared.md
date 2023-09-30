# undeclared dependenciec
## Cython
### 1.
**path**: `.repositories/scikit-learn/sklearn/_build_utils/__init__.py`
**line number**: 42
```python
    _check_cython_version()
    from Cython.Build import cythonize

    # Fast fail before cythonization if compiler fails compiling basic test
    # code even without OpenMP
    basic_check_build()


```
### 2.
**path**: `.repositories/scikit-learn/sklearn/_build_utils/__init__.py`
**line number**: 92
```python
    # Lazy import because cython is not a runtime dependency.
    from Cython import Tempita

    for template in templates:
        outfile = template.replace(".tp", "")

        # if the template is not updated, no need to output the cython file

```
### 3.
**path**: `.repositories/scikit-learn/sklearn/_build_utils/__init__.py`
**line number**: 27
```python
    try:
        import Cython
    except ModuleNotFoundError as e:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message) from e

    if parse(Cython.__version__) < parse(CYTHON_MIN_VERSION):

```
## array_api_compat
### 1.
**path**: `.repositories/scikit-learn/sklearn/utils/_array_api.py`
**line number**: 411
```python
    # message in case it is missing.
    import array_api_compat

    namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True

    # These namespaces need additional wrapping to smooth out small differences
    # between implementations

```
### 2.
**path**: `.repositories/scikit-learn/sklearn/utils/_array_api.py`
**line number**: 61
```python
        try:
            import array_api_compat  # noqa
        except ImportError:
            raise ImportError(
                "array_api_compat is required to dispatch arrays using the API"
                " specification"
            )

```
### 3.
**path**: `.repositories/scikit-learn/sklearn/utils/_testing.py`
**line number**: 1079
```python
    try:
        import array_api_compat  # noqa
    except ImportError:
        raise SkipTest(
            "array_api_compat is not installed: not checking array_api input"
        )


```
## cupy
### 1.
**path**: `.repositories/scikit-learn/sklearn/utils/_testing.py`
**line number**: 1113
```python
    elif array_namespace in {"cupy", "cupy.array_api"}:  # pragma: nocover
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise SkipTest("CuPy test requires cuda, which is not available")
    return xp, device, dtype

```
## setuptools
### 1.
**path**: `.repositories/scikit-learn/sklearn/_build_utils/pre_build_helpers.py`
**line number**: 10
```python

from setuptools.command.build_ext import customize_compiler, new_compiler


def compile_test_program(code, extra_preargs=None, extra_postargs=None):
    """Check that some C code can be compiled and run"""
    ccompiler = new_compiler()

```
## typing_extensions
### 1.
**path**: `.repositories/scikit-learn/sklearn/externals/_arff.py`
**line number**: 177
```python
    # typing_extensions is available when mypy is installed
    from typing_extensions import TypedDict

    class ArffContainerType(TypedDict):
        description: str
        relation: str
        attributes: List

```
