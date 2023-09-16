# undeclared dependenciec
## AppKit
### 1.
**path**: `.repositories/Kivy/kivy/core/spelling/spelling_osxappkit.py`
**line number**: 16
```python

from AppKit import NSSpellChecker, NSMakeRange

from kivy.core.spelling import SpellingBase, NoSuchLangError


class SpellingOSXAppKit(SpellingBase):

```
## Image
### 1.
**path**: `.repositories/Kivy/kivy/core/image/img_pil.py`
**line number**: 8
```python
try:
    import Image as PILImage
except ImportError:
    # for python3
    from PIL import Image as PILImage

from kivy.logger import Logger

```
## Leap
### 1.
**path**: `.repositories/Kivy/kivy/input/providers/leapfinger.py`
**line number**: 50
```python
        global Leap, InteractionBox
        import Leap
        from Leap import InteractionBox

        class LeapMotionListener(Leap.Listener):

            def on_init(self, controller):

```
### 2.
**path**: `.repositories/Kivy/kivy/input/providers/leapfinger.py`
**line number**: 51
```python
        import Leap
        from Leap import InteractionBox

        class LeapMotionListener(Leap.Listener):

            def on_init(self, controller):
                Logger.info('leapmotion: Initialized')

```
## PIL
### 1.
**path**: `.repositories/Kivy/kivy/core/image/img_pil.py`
**line number**: 11
```python
    # for python3
    from PIL import Image as PILImage

from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader

try:

```
### 2.
**path**: `.repositories/Kivy/kivy/tests/common.py`
**line number**: 331
```python
        from tkinter import Tk, Label, LEFT, RIGHT, BOTTOM, Button
        from PIL import Image, ImageTk

        self.retval = False

        root = Tk()


```
### 3.
**path**: `.repositories/Kivy/kivy/tests/common.py`
**line number**: 362
```python
        from tkinter import Tk, Label, LEFT, RIGHT, BOTTOM, Button
        from PIL import Image, ImageTk

        self.retval = False

        root = Tk()


```
### 4.
**path**: `.repositories/Kivy/kivy/core/text/text_pil.py`
**line number**: 7
```python

from PIL import Image, ImageFont, ImageDraw


from kivy.compat import text_type
from kivy.core.text import LabelBase
from kivy.core.image import ImageData

```
### 5.
**path**: `.repositories/Kivy/kivy/tools/generate-icons.py`
**line number**: 10
```python
import sys
from PIL import Image
from os.path import exists, join, realpath, basename, dirname
from os import makedirs
from argparse import ArgumentParser



```
### 6.
**path**: `.repositories/Kivy/kivy/tools/texturecompress.py`
**line number**: 33
```python
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink


```
### 7.
**path**: `.repositories/Kivy/kivy/atlas.py`
**line number**: 273
```python
        try:
            from PIL import Image
        except ImportError:
            Logger.critical('Atlas: Imaging/PIL are missing')
            raise

        if isinstance(size, (tuple, list)):

```
## PyInstaller
### 1.
**path**: `.repositories/Kivy/kivy/tools/packaging/pyinstaller_hooks/__init__.py`
**line number**: 81
```python
from kivy.factory import Factory
from PyInstaller.depend import bindepend

from os import environ
if 'KIVY_DOC' not in environ:
    from PyInstaller.utils.hooks import collect_submodules


```
### 2.
**path**: `.repositories/Kivy/kivy/tools/packaging/pyinstaller_hooks/__init__.py`
**line number**: 85
```python
if 'KIVY_DOC' not in environ:
    from PyInstaller.utils.hooks import collect_submodules

    curdir = dirname(__file__)

    kivy_modules = [
        'xml.etree.cElementTree',

```
### 3.
**path**: `.repositories/Kivy/kivy/tests/pyinstaller/test_pyinstaller.py`
**line number**: 12
```python
    try:
        import PyInstaller
    except ImportError:
        pytestmark = pytest.mark.skip("PyInstaller is not available")


@pytest.mark.incremental

```
## __main__
### 1.
**path**: `.repositories/Kivy/kivy/garden/__init__.py`
**line number**: 158
```python
    from os.path import join, dirname
    import __main__
    main_py_file = __main__.__file__
    garden_app_dir = join(dirname(main_py_file), 'libs', 'garden')


class GardenImporter(object):

```
## android
### 1.
**path**: `.repositories/Kivy/kivy/core/window/window_pygame.py`
**line number**: 31
```python
    if platform == 'android':
        import android
except ImportError:
    pass

# late binding
glReadPixels = GL_RGBA = GL_UNSIGNED_BYTE = None

```
### 2.
**path**: `.repositories/Kivy/kivy/app.py`
**line number**: 979
```python
        if platform == 'android':
            from android import mActivity
            mActivity.finishAndRemoveTask()
        else:
            self._stop()

    def _stop(self, *largs):

```
### 3.
**path**: `.repositories/Kivy/kivy/app.py`
**line number**: 1002
```python
        if platform == 'android':
            from android import mActivity
            mActivity.moveTaskToBack(True)
        else:
            Logger.info('App.pause() is not available on this OS.')

    def on_start(self):

```
### 4.
**path**: `.repositories/Kivy/kivy/base.py`
**line number**: 241
```python
        try:
            from android import remove_presplash
            remove_presplash()
        except ImportError:
            Logger.warning(
                'Base: Failed to import "android" module. '
                'Could not remove android presplash.')

```
### 5.
**path**: `.repositories/Kivy/kivy/core/audio/audio_pygame.py`
**line number**: 20
```python
        try:
            import android.mixer as mixer
        except ImportError:
            # old python-for-android version
            import android_mixer as mixer
    else:
        from pygame import mixer

```
### 6.
**path**: `.repositories/Kivy/kivy/core/window/__init__.py`
**line number**: 660
```python
        if not android:
            import android
        return android.get_keyboard_height()

    def _get_kivy_vkheight(self):
        mode = Config.get('kivy', 'keyboard_mode')
        if (

```
### 7.
**path**: `.repositories/Kivy/kivy/core/window/__init__.py`
**line number**: 2014
```python
        if key == 27 and platform == 'android':
            from android import mActivity
            mActivity.moveTaskToBack(True)
            return True
        elif WindowBase.on_keyboard.exit_on_escape:
            if key == 27 or all([is_osx, key in [113, 119], modifier == 1024]):
                if not self.dispatch('on_request_close', source='keyboard'):

```
### 8.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_android.py`
**line number**: 13
```python
from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
from android import python_act

AndroidString = autoclass('java.lang.String')
PythonActivity = python_act
Context = autoclass('android.content.Context')

```
### 9.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_android.py`
**line number**: 14
```python
from android.runnable import run_on_ui_thread
from android import python_act

AndroidString = autoclass('java.lang.String')
PythonActivity = python_act
Context = autoclass('android.content.Context')
VER = autoclass('android.os.Build$VERSION')

```
### 10.
**path**: `.repositories/Kivy/kivy/core/audio/audio_android.py`
**line number**: 8
```python
from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader


MediaPlayer = autoclass("android.media.MediaPlayer")
AudioManager = autoclass("android.media.AudioManager")

```
### 11.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 198
```python
            else:
                import android
                value = android.get_dpi()
        elif platform == 'ios':
            import ios
            value = ios.get_dpi()
        else:

```
### 12.
**path**: `.repositories/Kivy/kivy/input/providers/androidjoystick.py`
**line number**: 17
```python
try:
    import android  # NOQA
except ImportError:
    if 'KIVY_DOC' not in os.environ:
        raise Exception('android lib not found.')

from kivy.logger import Logger

```
### 13.
**path**: `.repositories/Kivy/kivy/core/window/window_sdl2.py`
**line number**: 256
```python
                        'WindowSDL: App stopped, on_pause() returned False.')
                    from android import mActivity
                    mActivity.finishAndRemoveTask()
                else:
                    Logger.info(
                        'WindowSDL: App doesn\'t support pause mode, stop.')
                    stopTouchApp()

```
### 14.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 72
```python
    try:
        import android
    except ImportError:
        print('Android lib is missing, cannot install android hooks')
        return

    from kivy.clock import Clock

```
## android_mixer
### 1.
**path**: `.repositories/Kivy/kivy/core/audio/audio_pygame.py`
**line number**: 23
```python
            # old python-for-android version
            import android_mixer as mixer
    else:
        from pygame import mixer
except:
    raise


```
## certifi
### 1.
**path**: `.repositories/Kivy/kivy/tests/test_urlrequest/test_urlrequest_requests.py`
**line number**: 10
```python

import certifi
import pytest
import responses
from kivy.network.urlrequest import UrlRequestRequests as UrlRequest
from requests.auth import HTTPBasicAuth
from responses import matchers

```
### 2.
**path**: `.repositories/Kivy/kivy/loader.py`
**line number**: 337
```python
                if platform in ['android', 'ios']:
                    import certifi
                    import ssl
                    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                    ssl_ctx.verify_mode = ssl.CERT_REQUIRED

                fd = urllib.request.urlopen(request, context=ssl_ctx)

```
### 3.
**path**: `.repositories/Kivy/kivy/tests/test_urlrequest/test_urlrequest_urllib.py`
**line number**: 137
```python
    """Passing a `ca_file` should not crash on http scheme, refs #6946"""
    import certifi
    obj = UrlRequestQueue([])
    queue = obj.queue
    req = UrlRequest(
        f"{scheme}://httpbin.org/get",
        on_success=obj._on_success,

```
### 4.
**path**: `.repositories/Kivy/kivy/network/urlrequest.py`
**line number**: 230
```python
        if platform in ['android', 'ios']:
            import certifi
            self.ca_file = ca_file or certifi.where()
        else:
            self.ca_file = ca_file

        #: Url of the request

```
## coverage
### 1.
**path**: `.repositories/Kivy/kivy/tests/test_coverage.py`
**line number**: 4
```python
try:
    import coverage
except ImportError:
    pytestmark = pytest.mark.skip("coverage not available")


kv_statement_lines = {4, 5, 6, 8, 9, 12, 15, 17}

```
## cv2
### 1.
**path**: `.repositories/Kivy/kivy/core/camera/camera_android.py`
**line number**: 188
```python
        import numpy as np
        from cv2 import cvtColor

        w, h = self._resolution
        arr = np.fromstring(buf, 'uint8').reshape((h + h / 2, w))
        arr = cvtColor(arr, 93)  # NV21 -> BGR
        return arr

```
### 2.
**path**: `.repositories/Kivy/kivy/core/camera/camera_opencv.py`
**line number**: 48
```python
    try:
        import cv2
        # here missing this OSX specific highgui thing.
        # I'm not on OSX so don't know if it is still valid in opencv >= 2
    except ImportError:
        raise


```
## dbus
### 1.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_dbusklipper.py`
**line number**: 14
```python
try:
    import dbus
    bus = dbus.SessionBus()
    proxy = bus.get_object("org.kde.klipper", "/klipper")
except:
    raise


```
## enchant
### 1.
**path**: `.repositories/Kivy/kivy/core/spelling/spelling_enchant.py`
**line number**: 12
```python

import enchant

from kivy.core.spelling import SpellingBase, NoSuchLangError
from kivy.compat import PY2



```
## ffmpeg
### 1.
**path**: `.repositories/Kivy/kivy/core/video/video_ffmpeg.py`
**line number**: 19
```python
try:
    import ffmpeg
except:
    raise

from kivy.core.video import VideoBase
from kivy.graphics.texture import Texture

```
## flask
### 1.
**path**: `.repositories/Kivy/kivy/modules/_webdebugger.py`
**line number**: 13
```python
try:
    from flask import Flask, render_template_string, make_response
except ImportError:
    Logger.error('WebDebugger: unable to import Flask. Install it!')
    raise

history_max = 250

```
## gi
### 1.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_gtk3.py`
**line number**: 14
```python

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)



```
### 2.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_gtk3.py`
**line number**: 16
```python
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)


class ClipboardGtk3(ClipboardBase):


```
### 3.
**path**: `.repositories/Kivy/kivy/core/camera/camera_gi.py`
**line number**: 10
```python

from gi.repository import Gst
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
from kivy.support import install_gobject_iteration
from kivy.logger import Logger

```
### 4.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 21
```python
    try:
        from gi.repository import GObject as gobject
    except ImportError:
        import gobject

    if hasattr(gobject, '_gobject_already_installed'):
        # already installed, don't do it twice.

```
## gimpfu
### 1.
**path**: `.repositories/Kivy/kivy/tools/image-testsuite/gimp28-testsuite.py`
**line number**: 6
```python
import random
from gimpfu import *

# Test suite configuration - key is test name, values are:
#
# alpha....: global alpha, used for all pixels except 't'
# patterns.: allowed v0 pattern characters (+ force include and exclude)

```
## gobject
### 1.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 23
```python
    except ImportError:
        import gobject

    if hasattr(gobject, '_gobject_already_installed'):
        # already installed, don't do it twice.
        return


```
## ios
### 1.
**path**: `.repositories/Kivy/kivy/core/window/__init__.py`
**line number**: 652
```python
    def _get_ios_kheight(self):
        import ios
        return ios.get_kheight()

    def _get_android_kheight(self):
        if USE_SDL2:  # Placeholder until the SDL2 bootstrap supports this
            return 0

```
### 2.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 201
```python
        elif platform == 'ios':
            import ios
            value = ios.get_dpi()
        else:
            # for all other platforms..
            from kivy.base import EventLoop
            EventLoop.ensure_window()

```
### 3.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 260
```python
        elif platform == 'ios':
            import ios
            value = ios.get_scale()
        elif platform in ('macosx', 'win'):
            value = self.dpi / 96.

        sync_pixel_scale(density=value)

```
## jnius
### 1.
**path**: `.repositories/Kivy/kivy/core/camera/camera_android.py`
**line number**: 1
```python
from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading

```
### 2.
**path**: `.repositories/Kivy/kivy/app.py`
**line number**: 853
```python
        elif platform == 'android':
            from jnius import autoclass, cast
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            context = cast('android.content.Context', PythonActivity.mActivity)
            file_p = cast('java.io.File', context.getFilesDir())
            data_dir = file_p.getAbsolutePath()
        elif platform == 'win':

```
### 3.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_android.py`
**line number**: 12
```python
from kivy.core.clipboard import ClipboardBase
from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
from android import python_act

AndroidString = autoclass('java.lang.String')
PythonActivity = python_act

```
### 4.
**path**: `.repositories/Kivy/kivy/core/audio/audio_android.py`
**line number**: 7
```python

from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader


MediaPlayer = autoclass("android.media.MediaPlayer")

```
### 5.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 256
```python
        if platform == 'android':
            import jnius
            Hardware = jnius.autoclass('org.renpy.android.Hardware')
            value = Hardware.metrics.scaledDensity
        elif platform == 'ios':
            import ios
            value = ios.get_scale()

```
### 6.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 291
```python
        if platform == 'android':
            from jnius import autoclass
            if USE_SDL2:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
            else:
                PythonActivity = autoclass('org.renpy.android.PythonActivity')
            config = PythonActivity.mActivity.getResources().getConfiguration()

```
### 7.
**path**: `.repositories/Kivy/kivy/metrics.py`
**line number**: 194
```python
            if USE_SDL2:
                import jnius
                Hardware = jnius.autoclass('org.renpy.android.Hardware')
                value = Hardware.getDPI()
            else:
                import android
                value = android.get_dpi()

```
## kivy_deps
### 1.
**path**: `.repositories/Kivy/kivy/tools/packaging/pyinstaller_hooks/__init__.py`
**line number**: 77
```python
try:
    import kivy_deps
except ImportError:
    kivy_deps = None
from kivy.factory import Factory
from PyInstaller.depend import bindepend


```
### 2.
**path**: `.repositories/Kivy/kivy/__init__.py`
**line number**: 304
```python
try:
    import kivy_deps
    for importer, modname, ispkg in pkgutil.iter_modules(kivy_deps.__path__):
        if not ispkg:
            continue
        if modname.startswith('gst'):
            _packages.insert(0, (importer, modname, 'kivy_deps'))

```
## mock
### 1.
**path**: `.repositories/Kivy/kivy/tests/test_utils.py`
**line number**: 11
```python
except:
    from mock import patch   # python 2.x

from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
        is_color_transparent, SafeList, get_random_color, get_hex_from_color,
        get_color_from_hex, strtotuple, QueryDict, intersection, difference,
        interpolate, _get_platform, deprecated, reify)

```
## numpy
### 1.
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
### 2.
**path**: `.repositories/Kivy/kivy/core/camera/camera_picamera.py`
**line number**: 19
```python
from picamera import PiCamera
import numpy


class CameraPiCamera(CameraBase):
    '''Implementation of CameraBase using PiCamera
    '''

```
## opencv
### 1.
**path**: `.repositories/Kivy/kivy/core/camera/camera_opencv.py`
**line number**: 21
```python
    # opencv 1 case
    import opencv as cv

    try:
        import opencv.highgui as hg
    except ImportError:
        class Hg(object):

```
### 2.
**path**: `.repositories/Kivy/kivy/core/camera/camera_opencv.py`
**line number**: 24
```python
    try:
        import opencv.highgui as hg
    except ImportError:
        class Hg(object):
            '''
            On OSX, not only are the import names different,
            but the API also differs.

```
## picamera
### 1.
**path**: `.repositories/Kivy/kivy/core/camera/camera_picamera.py`
**line number**: 18
```python

from picamera import PiCamera
import numpy


class CameraPiCamera(CameraBase):
    '''Implementation of CameraBase using PiCamera

```
## pygame
### 1.
**path**: `.repositories/Kivy/kivy/core/window/window_pygame.py`
**line number**: 13
```python
# fail early if possible
import pygame

from kivy.compat import PY2
from kivy.core.window import WindowBase
from kivy.core import CoreCriticalException
from os import environ

```
### 2.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_pygame.py`
**line number**: 20
```python
try:
    import pygame
    import pygame.scrap
except:
    raise



```
### 3.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_pygame.py`
**line number**: 21
```python
    import pygame
    import pygame.scrap
except:
    raise


class ClipboardPygame(ClipboardBase):

```
### 4.
**path**: `.repositories/Kivy/kivy/app.py`
**line number**: 1196
```python
        if platform == 'android' and not USE_SDL2:
            import pygame
            setting_key = pygame.K_MENU

        if key == setting_key:
            # toggle settings panel
            if not self.open_settings():

```
### 5.
**path**: `.repositories/Kivy/kivy/core/audio/audio_pygame.py`
**line number**: 25
```python
    else:
        from pygame import mixer
except:
    raise

# init pygame sound
mixer.pre_init(44100, -16, 2, 1024)

```
### 6.
**path**: `.repositories/Kivy/kivy/core/text/text_pygame.py`
**line number**: 18
```python
try:
    import pygame
except:
    raise

pygame_cache = {}
pygame_font_handles = {}

```
### 7.
**path**: `.repositories/Kivy/kivy/core/image/img_pygame.py`
**line number**: 18
```python
try:
    import pygame
except:
    raise


class ImageLoaderPygame(ImageLoaderBase):

```
### 8.
**path**: `.repositories/Kivy/kivy/input/providers/androidjoystick.py`
**line number**: 28
```python
if 'KIVY_DOC' not in os.environ:
    import pygame.joystick


class AndroidMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):

```
### 9.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 79
```python
    from kivy.logger import Logger
    import pygame

    Logger.info('Support: Android install hooks')

    # Init the library
    android.init()

```
## pyobjus
### 1.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_nspaste.py`
**line number**: 13
```python
try:
    from pyobjus import autoclass
    from pyobjus.dylib_manager import load_framework, INCLUDE
    load_framework(INCLUDE.AppKit)
except ImportError:
    raise SystemError('Pyobjus not installed. Please run the following'
        ' command to install it. `pip install --user pyobjus`')

```
### 2.
**path**: `.repositories/Kivy/kivy/core/clipboard/clipboard_nspaste.py`
**line number**: 14
```python
    from pyobjus import autoclass
    from pyobjus.dylib_manager import load_framework, INCLUDE
    load_framework(INCLUDE.AppKit)
except ImportError:
    raise SystemError('Pyobjus not installed. Please run the following'
        ' command to install it. `pip install --user pyobjus`')


```
### 3.
**path**: `.repositories/Kivy/kivy/core/audio/audio_avplayer.py`
**line number**: 9
```python
from kivy.core.audio import Sound, SoundLoader
from pyobjus import autoclass
from pyobjus.dylib_manager import load_framework, INCLUDE

load_framework(INCLUDE.AVFoundation)
AVAudioPlayer = autoclass("AVAudioPlayer")
NSURL = autoclass("NSURL")

```
### 4.
**path**: `.repositories/Kivy/kivy/core/audio/audio_avplayer.py`
**line number**: 10
```python
from pyobjus import autoclass
from pyobjus.dylib_manager import load_framework, INCLUDE

load_framework(INCLUDE.AVFoundation)
AVAudioPlayer = autoclass("AVAudioPlayer")
NSURL = autoclass("NSURL")
NSString = autoclass("NSString")

```
## pytest_trio
### 1.
**path**: `.repositories/Kivy/kivy/tests/common.py`
**line number**: 509
```python
                import trio
                from pytest_trio import trio_fixture
                func._force_trio_fixture = True
                return func
            except ImportError:
                return pytest.mark.skip(
                    reason='KIVY_EVENTLOOP == "trio" but '

```
## redis
### 1.
**path**: `.repositories/Kivy/kivy/storage/redisstore.py`
**line number**: 38
```python
if 'KIVY_DOC' not in os.environ:
    import redis


class RedisStore(AbstractStore):
    '''Store implementation using a Redis database.
    See the :mod:`kivy.storage` module documentation for more information.

```
### 2.
**path**: `.repositories/Kivy/kivy/tests/test_storage.py`
**line number**: 69
```python
            from kivy.storage.redisstore import RedisStore
            from redis.exceptions import ConnectionError
            try:
                params = dict(db=15)
                self._do_store_test_empty(RedisStore(params))
                self._do_store_test_filled(RedisStore(params))
            except ConnectionError:

```
## setuptools
### 1.
**path**: `.repositories/Kivy/kivy/tools/packaging/factory.py`
**line number**: 8
```python

from setuptools import Command

import kivy

ignore_list = (
    'kivy.lib',

```
## smb
### 1.
**path**: `.repositories/Kivy/kivy/loader.py`
**line number**: 312
```python
                # otherwise the data is occasionally not loaded
                from smb.SMBHandler import SMBHandler
            except ImportError:
                Logger.warning(
                    'Loader: can not load PySMB: make sure it is installed')
                return


```
## testsuite
### 1.
**path**: `.repositories/Kivy/kivy/tools/pep8checker/pep8.py`
**line number**: 2295
```python
    if options.doctest or options.testsuite:
        from testsuite.support import run_tests
        report = run_tests(style_guide)
    else:
        report = style_guide.check_files()

    if options.statistics:

```
## trio
### 1.
**path**: `.repositories/Kivy/kivy/tests/common.py`
**line number**: 508
```python
            try:
                import trio
                from pytest_trio import trio_fixture
                func._force_trio_fixture = True
                return func
            except ImportError:
                return pytest.mark.skip(

```
### 2.
**path**: `.repositories/Kivy/kivy/clock.py`
**line number**: 634
```python
        if lib == 'trio':
            import trio
            self._async_lib = trio

            async def wait_for(coro, t):
                with trio.move_on_after(t):
                    await coro

```
### 3.
**path**: `.repositories/Kivy/kivy/clock.py`
**line number**: 840
```python
        if lib == 'trio':
            import trio
            self._async_event = trio.Event()
            # we don't know if this is called after things have already been
            # scheduled, so don't delay for a full frame before processing
            # events
            self._async_event.set()

```
## twisted
### 1.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 170
```python

    import twisted

    # prevent installing more than once
    if hasattr(twisted, '_kivy_twisted_reactor_installed'):
        return
    twisted._kivy_twisted_reactor_installed = True

```
### 2.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 181
```python
    # install threaded-select reactor, to use with own event loop
    from twisted.internet import _threadedselect
    _threadedselect.install()

    # now we can import twisted reactor as usual
    from twisted.internet import reactor
    from twisted.internet.error import ReactorNotRunning

```
### 3.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 185
```python
    # now we can import twisted reactor as usual
    from twisted.internet import reactor
    from twisted.internet.error import ReactorNotRunning

    from collections import deque
    from kivy.base import EventLoop
    from kivy.logger import Logger

```
### 4.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 186
```python
    from twisted.internet import reactor
    from twisted.internet.error import ReactorNotRunning

    from collections import deque
    from kivy.base import EventLoop
    from kivy.logger import Logger
    from kivy.clock import Clock

```
### 5.
**path**: `.repositories/Kivy/kivy/support.py`
**line number**: 255
```python

    import twisted

    # prevent uninstalling more than once
    if not hasattr(twisted, '_kivy_twisted_reactor_installed'):
        return


```
## watchdog
### 1.
**path**: `.repositories/Kivy/kivy/tools/kviewer.py`
**line number**: 33
```python

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from os.path import dirname, basename, join


if len(argv) != 2:

```
### 2.
**path**: `.repositories/Kivy/kivy/tools/kviewer.py`
**line number**: 34
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from os.path import dirname, basename, join


if len(argv) != 2:
    print('usage: %s filename.kv' % argv[0])

```
## win32api
### 1.
**path**: `.repositories/Kivy/kivy/core/window/window_pygame.py`
**line number**: 248
```python

        import win32api
        import win32gui
        import win32con
        hwnd = pygame.display.get_wm_info()['window']
        icon_big = win32gui.LoadImage(
            None, filename, win32con.IMAGE_ICON,

```
## win32con
### 1.
**path**: `.repositories/Kivy/kivy/core/window/window_pygame.py`
**line number**: 250
```python
        import win32gui
        import win32con
        hwnd = pygame.display.get_wm_info()['window']
        icon_big = win32gui.LoadImage(
            None, filename, win32con.IMAGE_ICON,
            48, 48, win32con.LR_LOADFROMFILE)
        icon_small = win32gui.LoadImage(

```
### 2.
**path**: `.repositories/Kivy/kivy/core/window/window_sdl2.py`
**line number**: 340
```python
                    if Config.getboolean('graphics', 'resizable'):
                        import win32con
                        import ctypes
                        self._win.set_border_state(False)
                        # make windows dispatch,
                        # WM_NCCALCSIZE explicitly
                        ctypes.windll.user32.SetWindowPos(

```
## win32file
### 1.
**path**: `.repositories/Kivy/kivy/uix/filechooser.py`
**line number**: 122
```python
    try:
        from win32file import FILE_ATTRIBUTE_HIDDEN, GetFileAttributesExW, \
                              error
        _have_win32file = True
    except ImportError:
        Logger.error('filechooser: win32file module is missing')
        Logger.error('filechooser: we cannot check if a file is hidden or not')

```
## win32gui
### 1.
**path**: `.repositories/Kivy/kivy/core/window/window_pygame.py`
**line number**: 249
```python
        import win32api
        import win32gui
        import win32con
        hwnd = pygame.display.get_wm_info()['window']
        icon_big = win32gui.LoadImage(
            None, filename, win32con.IMAGE_ICON,
            48, 48, win32con.LR_LOADFROMFILE)

```
