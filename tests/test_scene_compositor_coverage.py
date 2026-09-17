"""
Extra coverage tests for core/scene_compositor.py.

Exercise branches NOT covered by test_scene_compositor.py: chroma-key /
border-flood fallback when DETR is unavailable, _segment_with_detr
happy/no-person/exception, _mask_coverage_plausible warn branches,
_get_panoptic_pipeline unavailable/cached/exception, _resolve_
segmentation_model_path except fallback, and compose_scene /
default_canvas_layout / validate_character_asset happy+fallback. All
model/IO is mocked; nothing loads a real model.
"""

import os
import sys
import tempfile
import warnings
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scene_compositor as sc


def _reset_pipeline_globals():
    sc._PANOPTIC_PIPELINE = None
    sc._PANOPTIC_AVAILABLE = None


def _reset_env():
    _reset_pipeline_globals()
    sc.SEGMENTATION_MODELS = {
        "detr_resnet_50_panoptic": "facebook/detr-resnet-50-panoptic"
    }
    sc.MODEL_PATHS = {"segmentation": "models/segmentation"}


def _solid_rgba(size=(64, 64), alpha=255, rgb=(120, 120, 120)):
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def _save_png(img, d, name="asset.png", **kwargs):
    p = os.path.join(d, name)
    img.save(p, **kwargs)
    return p


def _gradient_bg_asset(size=(128, 128)):
    h, w = size
    bg = np.linspace(60, 160, h, dtype=np.float32)[:, None] * np.ones(
        (1, w), dtype=np.float32
    )
    arr = np.stack([bg, bg, bg], axis=2).astype(np.uint8)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    blob = (
        (xx - w // 2) ** 2 / (0.18 * w) ** 2 + (yy - h // 2) ** 2 / (0.35 * h) ** 2
    ) <= 1
    arr = np.where(blob[..., None], (200, 40, 40), arr).astype(np.uint8)
    return Image.fromarray(arr)


class _MaskedPipe:
    def __init__(self, results=None, exc=None):
        self._results = results
        self._exc = exc
        self.called = 0

    def __call__(self, image):
        self.called += 1
        if self._exc is not None:
            raise self._exc
        return self._results or []


class _FakePILMask:
    def __init__(self, arr):
        self._arr = arr

    def convert(self, mode):
        a = self._arr.astype(np.uint8) * 255
        return Image.fromarray(a, mode=mode)


def _person_result(arr, label="person"):
    return {"label": label, "mask": _FakePILMask(arr)}


def _inband_blob(h=128, w=128):
    m = np.zeros((h, w), dtype=bool)
    m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return m


class TestResolveSegmentationModelPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_happy_returns_resolve_model_path_result(self):
        with patch(
            "generators.image_generator.resolve_model_path",
            return_value="/some/local/path",
        ):
            out = sc._resolve_segmentation_model_path()
        self.assertEqual(out, "/some/local/path")

    def test_except_fallback_local_missing(self):
        with tempfile.TemporaryDirectory() as d:
            miss = os.path.join(d, "no-such-seg")
            with patch(
                "generators.image_generator.resolve_model_path",
                side_effect=Exception("boom"),
            ), patch.dict(
                sc.MODEL_PATHS,
                {"segmentation": miss},
            ):
                out = sc._resolve_segmentation_model_path()
        self.assertEqual(out, "facebook/detr-resnet-50-panoptic")

    def test_except_fallback_local_exists(self):
        with tempfile.TemporaryDirectory() as d:
            segdir = os.path.join(d, "segmentation")
            os.makedirs(os.path.join(segdir, "detr_resnet_50_panoptic"))
            with patch(
                "generators.image_generator.resolve_model_path",
                side_effect=Exception("boom"),
            ), patch.dict(
                sc.MODEL_PATHS,
                {"segmentation": segdir},
            ):
                out = sc._resolve_segmentation_model_path()
            self.assertTrue(out.endswith("detr_resnet_50_panoptic"))
            self.assertIn(segdir, out)


class TestGetPanopticPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def setUp(self):
        _reset_pipeline_globals()

    def test_unavailable_short_circuits(self):
        sc._PANOPTIC_AVAILABLE = False
        self.assertIsNone(sc._get_panoptic_pipeline())

    def test_cached_pipeline_returned(self):
        sc._PANOPTIC_PIPELINE = _MaskedPipe()
        sc._PANOPTIC_AVAILABLE = True
        self.assertIs(sc._get_panoptic_pipeline(), sc._PANOPTIC_PIPELINE)

    def test_exception_marks_unavailable(self):
        fake_tf = types.ModuleType("transformers")

        def boom(*a, **k):
            raise Exception("no model")

        fake_tf.pipeline = boom
        sc._PANOPTIC_PIPELINE = None
        sc._PANOPTIC_AVAILABLE = None
        with patch.dict(sys.modules, {"transformers": fake_tf}), patch.object(
            sc, "_resolve_segmentation_model_path", return_value="/m"
        ), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = sc._get_panoptic_pipeline()
        self.assertIsNone(pipe)
        self.assertFalse(sc._PANOPTIC_AVAILABLE)

    def test_happy_caches_pipeline(self):
        fake = _MaskedPipe()
        fake_tf = types.ModuleType("transformers")
        fake_tf.pipeline = lambda *a, **k: fake
        with patch.dict(sys.modules, {"transformers": fake_tf}), patch.object(
            sc, "_resolve_segmentation_model_path", return_value="/m"
        ):
            sc._PANOPTIC_PIPELINE = None
            sc._PANOPTIC_AVAILABLE = None
            pipe = sc._get_panoptic_pipeline()
        self.assertIs(pipe, fake)
        self.assertTrue(sc._PANOPTIC_AVAILABLE)


class TestSegmentWithDetr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def setUp(self):
        _reset_pipeline_globals()

    def test_none_when_pipeline_unavailable(self):
        sc._PANOPTIC_AVAILABLE = False
        self.assertIsNone(sc._segment_with_detr(Image.new("RGB", (10, 10))))

    def test_happy_returns_person_mask(self):
        pipe = _MaskedPipe(
            results=[
                _person_result(np.ones((20, 20), dtype=bool), "person"),
                _person_result(np.zeros((20, 20), dtype=bool), "cat"),
            ]
        )
        sc._PANOPTIC_PIPELINE = pipe
        out = sc._segment_with_detr(Image.new("RGB", (20, 20)))
        self.assertIsNotNone(out)
        mask, method = out
        self.assertTrue(mask.sum() > 0)
        self.assertEqual(method, "detr_panoptic")
        self.assertEqual(pipe.called, 1)

    def test_no_person_returns_none(self):
        pipe = _MaskedPipe(
            results=[
                _person_result(np.ones((20, 20), dtype=bool), "cat"),
                {"label": "person", "mask": None},
            ]
        )
        sc._PANOPTIC_PIPELINE = pipe
        self.assertIsNone(sc._segment_with_detr(Image.new("RGB", (20, 20))))

    def test_exception_returns_none(self):
        pipe = _MaskedPipe(exc=Exception("inference boom"))
        sc._PANOPTIC_PIPELINE = pipe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(sc._segment_with_detr(Image.new("RGB", (20, 20))))

    def test_zero_pixel_best_returns_none(self):
        pipe = _MaskedPipe(results=[_person_result(np.zeros((20, 20), dtype=bool))])
        sc._PANOPTIC_PIPELINE = pipe
        self.assertIsNone(sc._segment_with_detr(Image.new("RGB", (20, 20))))


class TestMaskCoveragePlain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_empty_size_returns_zero(self):
        self.assertEqual(sc._mask_coverage(np.zeros((0, 0), dtype=bool)), 0.0)

    def test_too_small_warns(self):
        with warnings.catch_warnings(record=True) as c:
            warnings.simplefilter("always")
            self.assertFalse(
                sc._mask_coverage_plausible(np.zeros((100, 100), dtype=bool))
            )
        self.assertTrue(any("captured only" in str(w.message) for w in c))

    def test_too_large_warns(self):
        with warnings.catch_warnings(record=True) as c:
            warnings.simplefilter("always")
            self.assertFalse(
                sc._mask_coverage_plausible(np.ones((100, 100), dtype=bool))
            )
        self.assertTrue(any("left" in str(w.message) for w in c))

    def test_in_band_true(self):
        m = np.zeros((100, 100), dtype=bool)
        m[20:80, 20:80] = True
        self.assertTrue(sc._mask_coverage_plausible(m))


class TestMaskBbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_empty_returns_none(self):
        self.assertIsNone(sc._mask_bbox(np.zeros((10, 10), dtype=bool)))

    def test_nonempty_returns_bbox(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1, 2] = m[3, 4] = True
        self.assertEqual(sc._mask_bbox(m), (2, 1, 5, 4))


class TestSegmentCharacterChromaFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def setUp(self):
        sc._PANOPTIC_AVAILABLE = False
        sc._PANOPTIC_PIPELINE = None

    def test_chroma_key_fallback_success(self):
        with tempfile.TemporaryDirectory() as d:
            src = _save_png(_gradient_bg_asset(), d)
            blob = _inband_blob()
            with patch.object(
                sc,
                "_segment_with_chroma_key",
                return_value=(blob, "chroma_key"),
            ):
                mask_path, cutout_path, bbox, method = sc.segment_character(
                    src, output_dir=Path(d), name_hint="char"
                )
            self.assertEqual(method, "chroma_key")
            self.assertIsNotNone(cutout_path)
            self.assertTrue(os.path.exists(cutout_path))

    def test_retry_widens_band_accepts_large(self):
        with tempfile.TemporaryDirectory() as d:
            src = _save_png(_solid_rgba((64, 64), rgb=(120, 120, 120)), d)
            big = np.zeros((64, 64), dtype=bool)
            big[:, :51] = True
            with patch.object(
                sc,
                "_segment_with_chroma_key",
                return_value=(big, "chroma_key"),
            ):
                _, cut_retry, _, method = sc.segment_character(
                    src, output_dir=Path(d), name_hint="char", retry=True
                )
                _, cut_normal, _, _ = sc.segment_character(
                    src, output_dir=Path(d), name_hint="char", retry=False
                )
            self.assertIsNotNone(cut_retry)
            self.assertIsNone(cut_normal)
            self.assertEqual(method, "chroma_key")

    def test_output_dir_none(self):
        with tempfile.TemporaryDirectory() as d:
            src = _save_png(_gradient_bg_asset(), d)
            valid = _inband_blob()
            with patch.object(
                sc,
                "_segment_with_chroma_key",
                return_value=(valid, "chroma_key"),
            ):
                mask_path, cutout_path, bbox, method = sc.segment_character(
                    src, name_hint="c"
                )
        self.assertIsNone(mask_path)
        self.assertIsNone(cutout_path)
        self.assertIsNotNone(bbox)

    def test_nothing_usable_returns_none_method(self):
        with tempfile.TemporaryDirectory() as d:
            src = _save_png(_solid_rgba((64, 64), rgb=(120, 120, 120)), d)
            empty = np.zeros((64, 64), dtype=bool)
            with patch.object(
                sc,
                "_segment_with_chroma_key",
                return_value=(empty, "chroma_key"),
            ):
                mask_path, cutout_path, bbox, method = sc.segment_character(
                    src, name_hint="c"
                )
        self.assertIsNone(mask_path)
        self.assertIsNone(cutout_path)
        self.assertEqual(method, "none")


class TestComposeScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_default_anchor_and_scale(self):
        with tempfile.TemporaryDirectory() as d:
            bg = _save_png(
                _solid_rgba((64, 64), rgb=(30, 30, 30), alpha=255),
                d,
                "bg.png",
            )
            cut = _save_png(
                _solid_rgba((32, 32), rgb=(200, 40, 40), alpha=255),
                d,
                "cut.png",
            )
            canvas = sc.compose_scene(
                bg,
                [{"cutout_path": cut}],
                output_path=None,
            )
        self.assertEqual(canvas.size, (1024, 1024))
        self.assertEqual(canvas.mode, "RGBA")

    def test_explicit_params_and_resize_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            bg = os.path.join(d, "bg.png")
            cut = os.path.join(d, "cut.png")
            out = os.path.join(d, "out.png")
            Image.new("RGBA", (96, 48), (40, 40, 40, 255)).save(bg)
            cut_rgb = Image.new("RGB", (10, 20), (200, 40, 40))
            cut_rgb.save(cut)
            canvas = sc.compose_scene(
                bg,
                [
                    {
                        "cutout_path": cut,
                        "z": 2,
                        "scale": 0.5,
                        "anchor": (0.3, 0.4),
                    },
                ],
                canvas_width=200,
                canvas_height=150,
                output_path=out,
            )
            self.assertEqual(canvas.size, (200, 150))
            self.assertTrue(os.path.exists(out))


class TestDefaultCanvasLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_single(self):
        r = sc.default_canvas_layout(["A"])
        self.assertEqual(len(r["placements"]), 1)
        self.assertEqual(r["placements"][0]["anchor"][0], 0.5)
        self.assertEqual(r["placements"][0]["z"], 1)

    def test_pair(self):
        r = sc.default_canvas_layout(["A", "B"])
        xs = [p["anchor"][0] for p in r["placements"]]
        self.assertEqual(xs, [0.28, 0.72])

    def test_many(self):
        r = sc.default_canvas_layout(["A", "B", "C"])
        xs = [p["anchor"][0] for p in r["placements"]]
        self.assertEqual(xs, [1 / 4, 2 / 4, 3 / 4])


class TestValidateCharacterAsset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_env()

    def test_delegates_to_cutout(self):
        with tempfile.TemporaryDirectory() as d:
            p = _save_png(
                _solid_rgba((64, 64), alpha=255, rgb=(120, 120, 120)),
                d,
                "cut.png",
            )
            res = sc.validate_character_asset(p, name_hint="hero")
        self.assertEqual(res["name"], "hero")
        self.assertFalse(res["valid"])

    def test_rgba_convert_from_rgb(self):
        # A non-RGBA cutout must be converted (line 357) and flagged opaque.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rgb.png")
            Image.new("RGB", (40, 40), (150, 150, 150)).save(p)
            res = sc.validate_character_asset(p, name_hint="c")
        self.assertFalse(res["valid"])
        self.assertTrue(
            any("opaque" in i or "transparency" in i for i in res["issues"])
        )

    def test_fully_transparent_reports_empty(self):
        # A fully transparent cutout is an empty/degenerate asset (line 390).
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ft.png")
            Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(p)
            res = sc.validate_character_asset(p, name_hint="c")
        self.assertFalse(res["valid"])
        self.assertTrue(any("empty" in i or "transparent" in i for i in res["issues"]))


if __name__ == "__main__":
    unittest.main()
