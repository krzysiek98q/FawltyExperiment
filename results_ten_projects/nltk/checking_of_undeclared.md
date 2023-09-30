# undeclared dependenciec
## bllipparser
### 1.
**path**: `.repositories/nltk/nltk/parse/bllip.py`
**line number**: 87
```python
try:
    from bllipparser import RerankingParser
    from bllipparser.RerankingParser import get_unified_model_parameters

    def _ensure_bllip_import_or_error():
        pass


```
### 2.
**path**: `.repositories/nltk/nltk/parse/bllip.py`
**line number**: 88
```python
    from bllipparser import RerankingParser
    from bllipparser.RerankingParser import get_unified_model_parameters

    def _ensure_bllip_import_or_error():
        pass

except ImportError as ie:

```
## markdown_it
### 1.
**path**: `.repositories/nltk/nltk/corpus/reader/markdown.py`
**line number**: 111
```python
    def __init__(self, *args, parser=None, **kwargs):
        from markdown_it import MarkdownIt
        from mdit_plain.renderer import RendererPlain
        from mdit_py_plugins.front_matter import front_matter_plugin

        self.parser = parser
        if self.parser is None:

```
## mdit_plain
### 1.
**path**: `.repositories/nltk/nltk/corpus/reader/markdown.py`
**line number**: 112
```python
        from markdown_it import MarkdownIt
        from mdit_plain.renderer import RendererPlain
        from mdit_py_plugins.front_matter import front_matter_plugin

        self.parser = parser
        if self.parser is None:
            self.parser = MarkdownIt("commonmark", renderer_cls=RendererPlain)

```
## mdit_py_plugins
### 1.
**path**: `.repositories/nltk/nltk/corpus/reader/markdown.py`
**line number**: 113
```python
        from mdit_plain.renderer import RendererPlain
        from mdit_py_plugins.front_matter import front_matter_plugin

        self.parser = parser
        if self.parser is None:
            self.parser = MarkdownIt("commonmark", renderer_cls=RendererPlain)
            self.parser.use(front_matter_plugin)

```
## networkx
### 1.
**path**: `.repositories/nltk/nltk/parse/dependencygraph.py`
**line number**: 533
```python
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph."""
        import networkx

        nx_nodelist = list(range(1, len(self.nodes)))
        nx_edgelist = [
            (n, self._hd(n), self._rel(n)) for n in nx_nodelist if self._hd(n)
        ]

```
### 2.
**path**: `.repositories/nltk/nltk/parse/dependencygraph.py`
**line number**: 628
```python
        # currently doesn't work
        import networkx
        from matplotlib import pylab

        g = dg.nx_graph()
        g.info()
        pos = networkx.spring_layout(g, dim=1)

```
## norm
### 1.
**path**: `.repositories/nltk/nltk/translate/gale_church.py`
**line number**: 21
```python
try:
    from norm import logsf as norm_logsf
    from scipy.stats import norm
except ImportError:

    def erfcc(x):
        """Complementary error function."""

```
## numpypy
### 1.
**path**: `.repositories/nltk/nltk/__init__.py`
**line number**: 107
```python
try:
    import numpypy
except ImportError:
    pass

# Override missing methods on environments where it cannot be used like GAE.
import subprocess

```
## pandas
### 1.
**path**: `.repositories/nltk/nltk/test/twitter.ipynb`
**line number**: 1
```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [

```
## pygame
### 1.
**path**: `.repositories/nltk/nltk/corpus/reader/timit.py`
**line number**: 461
```python
            # FIXME: this won't work under python 3
            import pygame.mixer
            import StringIO

            pygame.mixer.init(16000)
            f = StringIO.StringIO(self.wav(utterance, start, end))
            pygame.mixer.Sound(f).play()

```
## pytest
### 1.
**path**: `.repositories/nltk/nltk/test/unit/test_rte_classify.py`
**line number**: 1
```python
import pytest

from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus


```
### 2.
**path**: `.repositories/nltk/nltk/test/gluesemantics_malt_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    from nltk.parse.malt import MaltParser

    try:
        depparser = MaltParser()

```
### 3.
**path**: `.repositories/nltk/nltk/test/conftest.py`
**line number**: 1
```python
import pytest

from nltk.corpus.reader import CorpusReader


@pytest.fixture(autouse=True)

```
### 4.
**path**: `.repositories/nltk/nltk/test/childes_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    import nltk.data

    try:
        nltk.data.find("corpora/childes/data-xml/Eng-USA-MOR/")

```
### 5.
**path**: `.repositories/nltk/nltk/test/probability_fixt.py`
**line number**: 6
```python
def setup_module():
    import pytest

    pytest.importorskip("numpy")

```
### 6.
**path**: `.repositories/nltk/nltk/test/unit/test_data.py`
**line number**: 1
```python
import pytest

import nltk.data


def test_find_raises_exception():

```
### 7.
**path**: `.repositories/nltk/nltk/test/unit/test_util.py`
**line number**: 1
```python
import pytest

from nltk.util import everygrams


@pytest.fixture

```
### 8.
**path**: `.repositories/nltk/nltk/test/unit/test_seekable_unicode_stream_reader.py`
**line number**: 4
```python

import pytest

from nltk.corpus.reader import SeekableUnicodeStreamReader


def check_reader(unicode_string, encoding):

```
### 9.
**path**: `.repositories/nltk/nltk/test/unit/test_corpora.py`
**line number**: 3
```python

import pytest

from nltk.corpus import (  # mwa_ppdb
    cess_cat,
    cess_esp,
    conll2007,

```
### 10.
**path**: `.repositories/nltk/nltk/test/gensim_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    pytest.importorskip("gensim")

```
### 11.
**path**: `.repositories/nltk/nltk/test/unit/test_json2csv_corpus.py`
**line number**: 14
```python

import pytest

from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities



```
### 12.
**path**: `.repositories/nltk/nltk/test/unit/test_tag.py`
**line number**: 21
```python
def setup_module(module):
    import pytest

    pytest.importorskip("numpy")

```
### 13.
**path**: `.repositories/nltk/nltk/test/setup_fixt.py`
**line number**: 7
```python
    Keyword arguments are passed to `nltk.internals.find_binary`."""
    import pytest

    try:
        find_binary(binary, **args)
    except LookupError:
        pytest.skip(f"Skipping test because the {binary} binary was not found.")

```
### 14.
**path**: `.repositories/nltk/nltk/test/setup_fixt.py`
**line number**: 22
```python
    """
    import pytest

    pytest.skip(
        "Skipping test because the doctests requiring jars are inconsistent on the CI."
    )

```
### 15.
**path**: `.repositories/nltk/nltk/test/unit/test_classify.py`
**line number**: 4
```python
"""
import pytest

from nltk import classify

TRAIN = [
    (dict(a=1, b=1, c=1), "y"),

```
### 16.
**path**: `.repositories/nltk/nltk/test/classify_fixt.py`
**line number**: 3
```python
def setup_module():
    import pytest

    pytest.importorskip("numpy")

```
### 17.
**path**: `.repositories/nltk/nltk/test/unit/test_bllip.py`
**line number**: 1
```python
import pytest

from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree


```
### 18.
**path**: `.repositories/nltk/nltk/test/unit/test_corenlp.py`
**line number**: 8
```python

import pytest

from nltk.parse import corenlp
from nltk.tree import Tree



```
### 19.
**path**: `.repositories/nltk/nltk/test/unit/test_distance.py`
**line number**: 3
```python

import pytest

from nltk.metrics.distance import edit_distance


class TestEditDistance:

```
### 20.
**path**: `.repositories/nltk/nltk/test/unit/test_tokenize.py`
**line number**: 7
```python

import pytest

from nltk.tokenize import (
    LegalitySyllableTokenizer,
    StanfordSegmenter,
    SyllableTokenizer,

```
### 21.
**path**: `.repositories/nltk/nltk/test/portuguese_en_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    pytest.skip("portuguese_en.doctest imports nltk.examples.pt which doesn't exist!")

```
### 22.
**path**: `.repositories/nltk/nltk/test/unit/test_twitter_auth.py`
**line number**: 7
```python

import pytest

pytest.importorskip("twython")

from nltk.twitter import Authenticate


```
### 23.
**path**: `.repositories/nltk/nltk/test/unit/lm/test_counter.py`
**line number**: 10
```python

import pytest

from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams


```
### 24.
**path**: `.repositories/nltk/nltk/test/unit/lm/test_models.py`
**line number**: 10
```python

import pytest

from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,

```
### 25.
**path**: `.repositories/nltk/nltk/test/unit/test_cfd_mutation.py`
**line number**: 3
```python

import pytest

from nltk import ConditionalFreqDist, tokenize


class TestEmptyCondFreq(unittest.TestCase):

```
### 26.
**path**: `.repositories/nltk/nltk/test/unit/test_hmm.py`
**line number**: 1
```python
import pytest

from nltk.tag import hmm


def _wikipedia_example_hmm():

```
## svgling
### 1.
**path**: `.repositories/nltk/nltk/tree/tree.py`
**line number**: 782
```python
    def _repr_svg_(self):
        from svgling import draw_tree

        return draw_tree(self)._repr_svg_()

    def __str__(self):
        return self.pformat()

```
## yaml
### 1.
**path**: `.repositories/nltk/nltk/data.py`
**line number**: 768
```python
    elif format == "yaml":
        import yaml

        resource_val = yaml.safe_load(opened_resource)
    else:
        # The resource is a text format.
        binary_data = opened_resource.read()

```
### 2.
**path**: `.repositories/nltk/nltk/corpus/reader/markdown.py`
**line number**: 207
```python
    def metadata_reader(self, stream):
        from yaml import safe_load

        return [
            safe_load(t.content)
            for t in self.parser.parse(stream.read())
            if t.type == "front_matter"

```
