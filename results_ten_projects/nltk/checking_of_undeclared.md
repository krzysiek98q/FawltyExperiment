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
## pycrfsuite
### 1.
**path**: `.repositories/nltk/nltk/tag/crf.py`
**line number**: 18
```python
try:
    import pycrfsuite
except ImportError:
    pass


class CRFTagger(TaggerI):

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
**path**: `.repositories/nltk/nltk/test/unit/test_seekable_unicode_stream_reader.py`
**line number**: 4
```python

import pytest

from nltk.corpus.reader import SeekableUnicodeStreamReader


def check_reader(unicode_string, encoding):

```
### 2.
**path**: `.repositories/nltk/nltk/test/conftest.py`
**line number**: 1
```python
import pytest

from nltk.corpus.reader import CorpusReader


@pytest.fixture(autouse=True)

```
### 3.
**path**: `.repositories/nltk/nltk/test/unit/test_rte_classify.py`
**line number**: 1
```python
import pytest

from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus


```
### 4.
**path**: `.repositories/nltk/nltk/test/unit/lm/test_counter.py`
**line number**: 10
```python

import pytest

from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams


```
### 5.
**path**: `.repositories/nltk/nltk/test/unit/test_tag.py`
**line number**: 21
```python
def setup_module(module):
    import pytest

    pytest.importorskip("numpy")

```
### 6.
**path**: `.repositories/nltk/nltk/test/unit/test_corpora.py`
**line number**: 3
```python

import pytest

from nltk.corpus import (  # mwa_ppdb
    cess_cat,
    cess_esp,
    conll2007,

```
### 7.
**path**: `.repositories/nltk/nltk/test/gensim_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    pytest.importorskip("gensim")

```
### 8.
**path**: `.repositories/nltk/nltk/test/gluesemantics_malt_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    from nltk.parse.malt import MaltParser

    try:
        depparser = MaltParser()

```
### 9.
**path**: `.repositories/nltk/nltk/test/classify_fixt.py`
**line number**: 3
```python
def setup_module():
    import pytest

    pytest.importorskip("numpy")

```
### 10.
**path**: `.repositories/nltk/nltk/test/childes_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    import nltk.data

    try:
        nltk.data.find("corpora/childes/data-xml/Eng-USA-MOR/")

```
### 11.
**path**: `.repositories/nltk/nltk/test/portuguese_en_fixt.py`
**line number**: 2
```python
def setup_module():
    import pytest

    pytest.skip("portuguese_en.doctest imports nltk.examples.pt which doesn't exist!")

```
### 12.
**path**: `.repositories/nltk/nltk/test/unit/test_data.py`
**line number**: 1
```python
import pytest

import nltk.data


def test_find_raises_exception():

```
### 13.
**path**: `.repositories/nltk/nltk/test/unit/test_json2csv_corpus.py`
**line number**: 14
```python

import pytest

from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities



```
### 14.
**path**: `.repositories/nltk/nltk/test/unit/test_distance.py`
**line number**: 3
```python

import pytest

from nltk.metrics.distance import edit_distance


class TestEditDistance:

```
### 15.
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
### 16.
**path**: `.repositories/nltk/nltk/test/setup_fixt.py`
**line number**: 22
```python
    """
    import pytest

    pytest.skip(
        "Skipping test because the doctests requiring jars are inconsistent on the CI."
    )

```
### 17.
**path**: `.repositories/nltk/nltk/test/unit/test_twitter_auth.py`
**line number**: 7
```python

import pytest

pytest.importorskip("twython")

from nltk.twitter import Authenticate


```
### 18.
**path**: `.repositories/nltk/nltk/test/unit/lm/test_models.py`
**line number**: 10
```python

import pytest

from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,

```
### 19.
**path**: `.repositories/nltk/nltk/test/unit/test_util.py`
**line number**: 1
```python
import pytest

from nltk.util import everygrams


@pytest.fixture

```
### 20.
**path**: `.repositories/nltk/nltk/test/unit/test_classify.py`
**line number**: 4
```python
"""
import pytest

from nltk import classify

TRAIN = [
    (dict(a=1, b=1, c=1), "y"),

```
### 21.
**path**: `.repositories/nltk/nltk/test/unit/test_tokenize.py`
**line number**: 7
```python

import pytest

from nltk.tokenize import (
    LegalitySyllableTokenizer,
    StanfordSegmenter,
    SyllableTokenizer,

```
### 22.
**path**: `.repositories/nltk/nltk/test/probability_fixt.py`
**line number**: 6
```python
def setup_module():
    import pytest

    pytest.importorskip("numpy")

```
### 23.
**path**: `.repositories/nltk/nltk/test/unit/test_corenlp.py`
**line number**: 8
```python

import pytest

from nltk.parse import corenlp
from nltk.tree import Tree



```
### 24.
**path**: `.repositories/nltk/nltk/test/unit/test_hmm.py`
**line number**: 1
```python
import pytest

from nltk.tag import hmm


def _wikipedia_example_hmm():

```
### 25.
**path**: `.repositories/nltk/nltk/test/unit/test_bllip.py`
**line number**: 1
```python
import pytest

from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree


```
### 26.
**path**: `.repositories/nltk/nltk/test/unit/test_cfd_mutation.py`
**line number**: 3
```python

import pytest

from nltk import ConditionalFreqDist, tokenize


class TestEmptyCondFreq(unittest.TestCase):

```
## sklearn
### 1.
**path**: `.repositories/nltk/nltk/classify/scikitlearn.py`
**line number**: 38
```python
try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    pass

__all__ = ["SklearnClassifier"]

```
### 2.
**path**: `.repositories/nltk/nltk/classify/scikitlearn.py`
**line number**: 39
```python
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    pass

__all__ = ["SklearnClassifier"]


```
### 3.
**path**: `.repositories/nltk/nltk/classify/scikitlearn.py`
**line number**: 124
```python
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB

    from nltk.classify.util import names_demo, names_demo_features

    # Bernoulli Naive Bayes is designed for binary classification. We set the

```
### 4.
**path**: `.repositories/nltk/nltk/classify/scikitlearn.py`
**line number**: 125
```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB

    from nltk.classify.util import names_demo, names_demo_features

    # Bernoulli Naive Bayes is designed for binary classification. We set the
    # binarize option to False since we know we're passing boolean features.

```
### 5.
**path**: `.repositories/nltk/nltk/parse/transitionparser.py`
**line number**: 18
```python
    from scipy import sparse
    from sklearn import svm
    from sklearn.datasets import load_svmlight_file
except ImportError:
    pass

from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI

```
### 6.
**path**: `.repositories/nltk/nltk/parse/transitionparser.py`
**line number**: 19
```python
    from sklearn import svm
    from sklearn.datasets import load_svmlight_file
except ImportError:
    pass

from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI


```
### 7.
**path**: `.repositories/nltk/nltk/sentiment/util.py`
**line number**: 871
```python
if __name__ == "__main__":
    from sklearn.svm import LinearSVC

    from nltk.classify import MaxentClassifier, NaiveBayesClassifier
    from nltk.classify.scikitlearn import SklearnClassifier
    from nltk.twitter.common import _outf_writer, extract_fields


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
