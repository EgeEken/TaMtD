from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JXL_DEC_ERROR = 1
JXL_DEC_SUCCESS = 0
JXL_DEC_NEED_IMAGE_OUT_BUFFER = 5
JXL_DEC_BASIC_INFO = 0x40
JXL_DEC_FRAME = 0x400
JXL_DEC_FULL_IMAGE = 0x1000
JXL_DEC_FRAME_PROGRESSION = 0x8000
JXL_DEC_NEED_MORE_INPUT = 2
JXL_DEC_NEED_PREVIEW_OUT_BUFFER = 4
JXL_DEC_BOX = 0x4000
JXL_DEC_BOX_COMPLETE = 0x10000
JXL_TYPE_UINT8 = 2


@dataclass(frozen=True)
class ProgressiveImage:
    ratio: int
    pixels: np.ndarray | None
    decoder_ms: float


@dataclass(frozen=True)
class JxlResult:
    rgb: np.ndarray
    progressions: tuple[ProgressiveImage, ...]
    statuses: tuple[int, ...]
    total_ms: float


class JxlPixelFormat(ctypes.Structure):
    _fields_ = [
        ("num_channels", ctypes.c_uint32),
        ("data_type", ctypes.c_int),
        ("endianness", ctypes.c_int),
        ("align", ctypes.c_size_t),
    ]


class AvifImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("yuv_format", ctypes.c_int),
        ("yuv_range", ctypes.c_int),
        ("yuv_chroma_sample_position", ctypes.c_int),
        ("yuv_planes", ctypes.POINTER(ctypes.c_uint8) * 3),
        ("yuv_row_bytes", ctypes.c_uint32 * 3),
        ("image_owns_yuv_planes", ctypes.c_int),
    ]


class AvifRgbImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("format", ctypes.c_int),
        ("chroma_upsampling", ctypes.c_int),
        ("chroma_downsampling", ctypes.c_int),
        ("avoid_libyuv", ctypes.c_int),
        ("ignore_alpha", ctypes.c_int),
        ("alpha_premultiplied", ctypes.c_int),
        ("is_float", ctypes.c_int),
        ("max_threads", ctypes.c_int),
        ("pixels", ctypes.POINTER(ctypes.c_uint8)),
        ("row_bytes", ctypes.c_uint32),
    ]


class CodecLibraries:
    def __init__(self, build_root: str | Path):
        self.build_root = Path(build_root).resolve()
        self._dll_handles = []
        if os.name == "nt":
            for directory in (
                self.build_root / "build-libaom",
                self.build_root / "build-libavif",
                self.build_root / "build-libwebp",
                self.build_root / "build-libjxl" / "lib",
                self.build_root / "build-libjxl" / "third_party" / "brotli",
                Path(os.environ.get("TAMTD_PHASE47_COMPILER_BIN", r"C:\Strawberry\c\bin")),
            ):
                if directory.is_dir():
                    self._dll_handles.append(os.add_dll_directory(str(directory)))
        self.aom = ctypes.CDLL(str(self._find("build-libaom", "libaom.dll", "libaom.so", "libaom.dylib")))
        self.avif = ctypes.CDLL(str(self._find("build-libavif", "libavif.dll", "libavif.so", "libavif.dylib")))
        self.webp = ctypes.CDLL(str(self._find("build-libwebp", "libwebp.dll", "libwebp.so", "libwebp.dylib")))
        self.jxl = ctypes.CDLL(str(self._find("build-libjxl/lib", "libjxl.dll", "libjxl.so", "libjxl.dylib")))
        self.jxl_threads = ctypes.CDLL(str(self._find("build-libjxl/lib", "libjxl_threads.dll", "libjxl_threads.so", "libjxl_threads.dylib")))
        self._configure_av1()
        self._configure_avif()
        self._configure_webp()
        self._configure_jxl()

    def _find(self, relative: str, *names: str) -> Path:
        base = self.build_root / relative
        for name in names:
            path = base / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"required Phase 4.7 decoder binary missing in {base}")

    def _configure_av1(self) -> None:
        callback_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self._av1_callback_type = callback_type
        self.aom.aom_phase47_set_stage_callback.argtypes = [callback_type, ctypes.c_void_p]
        self.aom.aom_phase47_set_stage_callback.restype = None
        self.aom.aom_phase47_set_parse_only.argtypes = [ctypes.c_int]
        self.aom.aom_phase47_set_parse_only.restype = None

    def _configure_avif(self) -> None:
        self.avif.avifDecoderCreate.argtypes = []
        self.avif.avifDecoderCreate.restype = ctypes.c_void_p
        self.avif.avifDecoderDestroy.argtypes = [ctypes.c_void_p]
        self.avif.avifDecoderDestroy.restype = None
        self.avif.avifDecoderReadMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.avif.avifDecoderReadMemory.restype = ctypes.c_int
        self.avif.avifImageCreateEmpty.argtypes = []
        self.avif.avifImageCreateEmpty.restype = ctypes.POINTER(AvifImage)
        self.avif.avifImageDestroy.argtypes = [ctypes.POINTER(AvifImage)]
        self.avif.avifImageDestroy.restype = None
        self.avif.avifRGBImageSetDefaults.argtypes = [ctypes.POINTER(AvifRgbImage), ctypes.POINTER(AvifImage)]
        self.avif.avifRGBImageSetDefaults.restype = None
        self.avif.avifRGBImageAllocatePixels.argtypes = [ctypes.POINTER(AvifRgbImage)]
        self.avif.avifRGBImageAllocatePixels.restype = ctypes.c_int
        self.avif.avifRGBImageFreePixels.argtypes = [ctypes.POINTER(AvifRgbImage)]
        self.avif.avifRGBImageFreePixels.restype = None
        self.avif.avifImageYUVToRGB.argtypes = [ctypes.POINTER(AvifImage), ctypes.POINTER(AvifRgbImage)]
        self.avif.avifImageYUVToRGB.restype = ctypes.c_int

    def _configure_webp(self) -> None:
        self._webp_callback_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.webp.WebPPhase47SetRowCallback.argtypes = [self._webp_callback_type, ctypes.c_void_p]
        self.webp.WebPPhase47SetRowCallback.restype = None
        self.webp.WebPDecodeRGB.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.webp.WebPDecodeRGB.restype = ctypes.POINTER(ctypes.c_uint8)
        self.webp.WebPDecodeYUV.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self.webp.WebPDecodeYUV.restype = ctypes.POINTER(ctypes.c_uint8)
        self.webp.WebPFree.argtypes = [ctypes.c_void_p]
        self.webp.WebPFree.restype = None

    def _configure_jxl(self) -> None:
        self.jxl.JxlDecoderSetParallelRunner.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.jxl.JxlDecoderSetParallelRunner.restype = ctypes.c_int
        self.jxl.JxlDecoderCreate.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderCreate.restype = ctypes.c_void_p
        self.jxl.JxlDecoderDestroy.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderDestroy.restype = None
        self.jxl.JxlDecoderSubscribeEvents.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.jxl.JxlDecoderSubscribeEvents.restype = ctypes.c_int
        self.jxl.JxlDecoderSetProgressiveDetail.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.jxl.JxlDecoderSetProgressiveDetail.restype = ctypes.c_int
        self.jxl.JxlDecoderSetInput.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        self.jxl.JxlDecoderSetInput.restype = ctypes.c_int
        self.jxl.JxlDecoderReleaseInput.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderReleaseInput.restype = ctypes.c_size_t
        self.jxl.JxlDecoderCloseInput.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderCloseInput.restype = None
        self.jxl.JxlDecoderProcessInput.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderProcessInput.restype = ctypes.c_int
        self.jxl.JxlDecoderImageOutBufferSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(JxlPixelFormat), ctypes.POINTER(ctypes.c_size_t)]
        self.jxl.JxlDecoderImageOutBufferSize.restype = ctypes.c_int
        self.jxl.JxlDecoderSetImageOutBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(JxlPixelFormat), ctypes.c_void_p, ctypes.c_size_t]
        self.jxl.JxlDecoderSetImageOutBuffer.restype = ctypes.c_int
        self.jxl.JxlDecoderFlushImage.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderFlushImage.restype = ctypes.c_int
        self.jxl.JxlDecoderGetIntendedDownsamplingRatio.argtypes = [ctypes.c_void_p]
        self.jxl.JxlDecoderGetIntendedDownsamplingRatio.restype = ctypes.c_size_t
        self.jxl_threads.JxlResizableParallelRunner.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
        ]
        self.jxl_threads.JxlResizableParallelRunner.restype = ctypes.c_int
        self.jxl_threads.JxlResizableParallelRunnerCreate.argtypes = [ctypes.c_void_p]
        self.jxl_threads.JxlResizableParallelRunnerCreate.restype = ctypes.c_void_p
        self.jxl_threads.JxlResizableParallelRunnerSetThreads.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.jxl_threads.JxlResizableParallelRunnerSetThreads.restype = None
        self.jxl_threads.JxlResizableParallelRunnerDestroy.argtypes = [ctypes.c_void_p]
        self.jxl_threads.JxlResizableParallelRunnerDestroy.restype = None

    def decode_jxl(
        self,
        payload: bytes,
        stop_after_progression: int | None = None,
        capture_progression: bool = True,
        capture_progressive_pixels: bool = True,
    ) -> JxlResult:
        decoder = self.jxl.JxlDecoderCreate(None)
        if not decoder:
            raise RuntimeError("JxlDecoderCreate failed")
        runner = self.jxl_threads.JxlResizableParallelRunnerCreate(None)
        if not runner:
            self.jxl.JxlDecoderDestroy(decoder)
            raise RuntimeError("JxlResizableParallelRunnerCreate failed")
        start = time.perf_counter_ns()
        try:
            parallel_runner = ctypes.cast(self.jxl_threads.JxlResizableParallelRunner, ctypes.c_void_p)
            if self.jxl.JxlDecoderSetParallelRunner(decoder, parallel_runner, runner) != JXL_DEC_SUCCESS:
                raise RuntimeError("JxlDecoderSetParallelRunner failed")
            events = JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE
            if capture_progression:
                events |= JXL_DEC_FRAME_PROGRESSION
            if self.jxl.JxlDecoderSubscribeEvents(decoder, events) != JXL_DEC_SUCCESS:
                raise RuntimeError("JxlDecoderSubscribeEvents failed")
            input_buffer = ctypes.create_string_buffer(payload)
            input_pointer = ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_uint8))
            if self.jxl.JxlDecoderSetInput(decoder, input_pointer, len(payload)) != JXL_DEC_SUCCESS:
                raise RuntimeError("JxlDecoderSetInput failed")
            self.jxl.JxlDecoderCloseInput(decoder)
            fmt = JxlPixelFormat(3, JXL_TYPE_UINT8, 0, 0)
            output_size = ctypes.c_size_t()
            output_buffer = None
            rgb = None
            status_history = []
            progressions: list[ProgressiveImage] = []
            while True:
                status = self.jxl.JxlDecoderProcessInput(decoder)
                status_history.append(int(status))
                if status == JXL_DEC_BASIC_INFO:
                    self.jxl_threads.JxlResizableParallelRunnerSetThreads(runner, 1)
                    if self.jxl.JxlDecoderImageOutBufferSize(decoder, ctypes.byref(fmt), ctypes.byref(output_size)) != JXL_DEC_SUCCESS:
                        raise RuntimeError("JxlDecoderImageOutBufferSize failed")
                    if output_size.value != 256 * 256 * 3:
                        raise ValueError(f"unexpected JXL RGB buffer size: {output_size.value}")
                elif status == JXL_DEC_FRAME:
                    continue
                elif status == JXL_DEC_NEED_IMAGE_OUT_BUFFER:
                    if output_size.value == 0:
                        raise RuntimeError("JXL requested an output buffer before basic info")
                    output_buffer = ctypes.create_string_buffer(output_size.value)
                    if self.jxl.JxlDecoderSetImageOutBuffer(decoder, ctypes.byref(fmt), output_buffer, output_size.value) != JXL_DEC_SUCCESS:
                        raise RuntimeError("JxlDecoderSetImageOutBuffer failed")
                elif status == JXL_DEC_FRAME_PROGRESSION:
                    ratio = int(self.jxl.JxlDecoderGetIntendedDownsamplingRatio(decoder))
                    if self.jxl.JxlDecoderFlushImage(decoder) != JXL_DEC_SUCCESS:
                        raise RuntimeError("JxlDecoderFlushImage failed at a progression event")
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                    image = None
                    if capture_progressive_pixels and not progressions:
                        image = np.frombuffer(output_buffer, dtype=np.uint8, count=256 * 256 * 3).reshape(256, 256, 3).copy()
                    progressions.append(ProgressiveImage(ratio, image, elapsed_ms))
                    if stop_after_progression is not None and len(progressions) >= stop_after_progression:
                        if image is None:
                            raise RuntimeError("an early JXL progression stop requires capturing the decoder output")
                        rgb = image
                        break
                elif status == JXL_DEC_FULL_IMAGE:
                    if output_buffer is None:
                        raise RuntimeError("JXL full image event arrived without an output buffer")
                    rgb = np.frombuffer(output_buffer, dtype=np.uint8, count=256 * 256 * 3).reshape(256, 256, 3).copy()
                elif status == JXL_DEC_SUCCESS:
                    if rgb is None:
                        raise RuntimeError("JXL decoder completed without full image output")
                    break
                elif status == JXL_DEC_NEED_MORE_INPUT:
                    raise RuntimeError("JPEG XL decoder requested more input after the complete stream was supplied")
                elif status in (JXL_DEC_ERROR, JXL_DEC_NEED_PREVIEW_OUT_BUFFER, JXL_DEC_BOX, JXL_DEC_BOX_COMPLETE):
                    raise RuntimeError(f"unexpected JXL decoder event/status {status}")
                else:
                    raise RuntimeError(f"unexpected JXL decoder status {status}")
            return JxlResult(rgb, tuple(progressions), tuple(status_history), (time.perf_counter_ns() - start) / 1_000_000)
        finally:
            self.jxl.JxlDecoderDestroy(decoder)
            self.jxl_threads.JxlResizableParallelRunnerDestroy(runner)

    def decode_webp_coefficients(self, payload: bytes) -> tuple[np.ndarray, float, float]:
        image_buffer = np.empty(16 * 16 * 800, dtype=np.uint8)
        qmatrix_buffer = np.empty(4 * 8 * 4, dtype=np.uint8)
        callback_state = {"rows": 0, "boundary_ns": None, "copy_ns": 0, "width": 0, "height": 0, "mb_w": 0}

        def on_row(width, height, mb_y, mb_w, row_bytes, row_data, quant_bytes, quant_data, _context):
            before = time.perf_counter_ns()
            if width != 256 or height != 256 or mb_w != 16 or row_bytes != mb_w * 800 or quant_bytes != qmatrix_buffer.nbytes:
                raise ValueError("WebP patch callback geometry or native struct size changed")
            row = ctypes.string_at(row_data, row_bytes)
            image_buffer[mb_y * row_bytes:(mb_y + 1) * row_bytes] = np.frombuffer(row, dtype=np.uint8)
            if mb_y == 0:
                qmatrix_buffer[:] = np.frombuffer(ctypes.string_at(quant_data, quant_bytes), dtype=np.uint8)
            callback_state["rows"] += 1
            callback_state["width"] = width
            callback_state["height"] = height
            callback_state["mb_w"] = mb_w
            callback_state["copy_ns"] += time.perf_counter_ns() - before
            if mb_y == 15:
                callback_state["boundary_ns"] = time.perf_counter_ns()

        callback = self._webp_callback_type(on_row)
        self.webp.WebPPhase47SetRowCallback(callback, None)
        image = ctypes.create_string_buffer(payload)
        width = ctypes.c_int()
        height = ctypes.c_int()
        start = time.perf_counter_ns()
        try:
            pointer = self.webp.WebPDecodeRGB(ctypes.cast(image, ctypes.POINTER(ctypes.c_uint8)), len(payload), ctypes.byref(width), ctypes.byref(height))
        finally:
            self.webp.WebPPhase47SetRowCallback(self._webp_callback_type(), None)
        if pointer:
            self.webp.WebPFree(pointer)
            raise RuntimeError("WebP parse-only hook unexpectedly returned reconstructed RGB")
        if callback_state["rows"] != 16 or callback_state["boundary_ns"] is None:
            raise RuntimeError(f"WebP coefficient hook saw {callback_state['rows']} rows")
        features = _webp_rows_to_features(image_buffer, qmatrix_buffer)
        decoder_ms = (callback_state["boundary_ns"] - start) / 1_000_000
        adapter_ms = max(0.0, (time.perf_counter_ns() - callback_state["boundary_ns"]) / 1_000_000)
        return features, decoder_ms, adapter_ms

    def decode_webp_yuv(self, payload: bytes) -> tuple[int, int, float]:
        image = ctypes.create_string_buffer(payload)
        y = ctypes.POINTER(ctypes.c_uint8)()
        u = ctypes.POINTER(ctypes.c_uint8)()
        v = ctypes.POINTER(ctypes.c_uint8)()
        y_stride = ctypes.c_int()
        uv_stride = ctypes.c_int()
        width = ctypes.c_int()
        height = ctypes.c_int()
        start = time.perf_counter_ns()
        result = self.webp.WebPDecodeYUV(
            ctypes.cast(image, ctypes.POINTER(ctypes.c_uint8)), len(payload),
            ctypes.byref(width), ctypes.byref(height),
            ctypes.byref(u), ctypes.byref(v), ctypes.byref(y_stride), ctypes.byref(uv_stride),
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        if not result or width.value != 256 or height.value != 256:
            raise RuntimeError("WebP full YUV decode failed or returned unexpected dimensions")
        self.webp.WebPFree(result)
        return width.value, height.value, elapsed_ms

    def decode_webp_rgb(self, payload: bytes) -> tuple[np.ndarray, float]:
        image = ctypes.create_string_buffer(payload)
        width = ctypes.c_int()
        height = ctypes.c_int()
        start = time.perf_counter_ns()
        pointer = self.webp.WebPDecodeRGB(
            ctypes.cast(image, ctypes.POINTER(ctypes.c_uint8)), len(payload), ctypes.byref(width), ctypes.byref(height)
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        if not pointer or width.value != 256 or height.value != 256:
            raise RuntimeError("WebP full RGB decode failed or returned unexpected dimensions")
        pixels = np.ctypeslib.as_array(pointer, shape=(256 * 256 * 3,)).reshape(256, 256, 3).copy()
        self.webp.WebPFree(pointer)
        return pixels, elapsed_ms

    def decode_avif_stage(self, payload: bytes) -> tuple[np.ndarray, float, float]:
        callback_state = {"features": None, "shape": None, "timestamp_ns": None, "feature_ready_ns": None}

        def on_stage(width, height, grid_width, grid_height, features, feature_count, _context):
            callback_state["timestamp_ns"] = time.perf_counter_ns()
            if (width, height, grid_width, grid_height, feature_count) != (256, 256, 16, 16, 16 * 16 * 62):
                raise ValueError("AV1 instrumentation callback schema or geometry changed")
            callback_state["shape"] = (grid_height * grid_width, 62)
            callback_state["features"] = np.ctypeslib.as_array(features, shape=(feature_count,)).reshape(256, 62).copy()
            callback_state["feature_ready_ns"] = time.perf_counter_ns()

        callback = self._av1_callback_type(on_stage)
        self.aom.aom_phase47_set_stage_callback(callback, None)
        self.aom.aom_phase47_set_parse_only(1)
        decoder = self.avif.avifDecoderCreate()
        image = self.avif.avifImageCreateEmpty()
        if not decoder or not image:
            self.aom.aom_phase47_set_parse_only(0)
            self.aom.aom_phase47_set_stage_callback(self._av1_callback_type(), None)
            raise RuntimeError("libavif decoder/image allocation failed")
        buffer = ctypes.create_string_buffer(payload)
        start = time.perf_counter_ns()
        try:
            result = self.avif.avifDecoderReadMemory(
                decoder, image, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), len(payload)
            )
        finally:
            self.aom.aom_phase47_set_parse_only(0)
            self.aom.aom_phase47_set_stage_callback(self._av1_callback_type(), None)
            self.avif.avifImageDestroy(image)
            self.avif.avifDecoderDestroy(decoder)
        if result != 0 or callback_state["features"] is None or callback_state["timestamp_ns"] is None:
            raise RuntimeError(f"AV1 pre-reconstruction decode failed (avifResult={result})")
        decoder_ms = (callback_state["timestamp_ns"] - start) / 1_000_000
        adapter_ms = max(0.0, (callback_state["feature_ready_ns"] - callback_state["timestamp_ns"]) / 1_000_000)
        return callback_state["features"], decoder_ms, adapter_ms

    def decode_avif_yuv(self, payload: bytes) -> tuple[AvifImage, float]:
        decoder = self.avif.avifDecoderCreate()
        image = self.avif.avifImageCreateEmpty()
        buffer = ctypes.create_string_buffer(payload)
        if not decoder or not image:
            raise RuntimeError("libavif decoder/image allocation failed")
        start = time.perf_counter_ns()
        result = self.avif.avifDecoderReadMemory(
            decoder, image, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), len(payload)
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.avif.avifDecoderDestroy(decoder)
        if result != 0 or image.contents.width != 256 or image.contents.height != 256:
            self.avif.avifImageDestroy(image)
            raise RuntimeError(f"AVIF full YUV decode failed (avifResult={result})")
        return image, elapsed_ms

    def decode_avif_rgb(self, payload: bytes) -> tuple[np.ndarray, float, float]:
        image, yuv_ms = self.decode_avif_yuv(payload)
        rgb = AvifRgbImage()
        self.avif.avifRGBImageSetDefaults(ctypes.byref(rgb), image)
        rgb.format = 0
        rgb.max_threads = 1
        if rgb.depth != 8 or self.avif.avifRGBImageAllocatePixels(ctypes.byref(rgb)) != 0:
            self.avif.avifImageDestroy(image)
            raise RuntimeError("libavif RGB output allocation failed")
        start = time.perf_counter_ns()
        result = self.avif.avifImageYUVToRGB(image, ctypes.byref(rgb))
        rgb_ms = (time.perf_counter_ns() - start) / 1_000_000
        if result != 0:
            self.avif.avifRGBImageFreePixels(ctypes.byref(rgb))
            self.avif.avifImageDestroy(image)
            raise RuntimeError(f"libavif YUV-to-RGB conversion failed (avifResult={result})")
        raw = np.ctypeslib.as_array(rgb.pixels, shape=(rgb.row_bytes * rgb.height,)).reshape(rgb.height, rgb.row_bytes)
        pixels = raw[:, : rgb.width * 3].reshape(rgb.height, rgb.width, 3).copy()
        self.avif.avifRGBImageFreePixels(ctypes.byref(rgb))
        self.avif.avifImageDestroy(image)
        return pixels, yuv_ms, rgb_ms


WEBP_MODE_MAP = np.asarray([0, 7, 1, 2, 3, 3, 3, 3, 3, 3], dtype=np.int64)
WEBP_CHROMA_MODE_MAP = np.asarray([0, 7, 1, 2], dtype=np.int64)
WEBP_LOW_FREQUENCY_POSITIONS = np.asarray([0, 1, 4, 2, 8, 5, 6, 9], dtype=np.int64)


def _webp_rows_to_features(rows: np.ndarray, quant: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.uint8).reshape(16, 16, 800)
    quant = np.frombuffer(quant.tobytes(), dtype="<i4").reshape(4, 8)
    features = np.zeros((256, 62), dtype=np.float32)
    coeffs = np.empty((24, 16), dtype="<i2")
    for y in range(16):
        for x in range(16):
            token = y * 16 + x
            block = rows[y, x]
            coeffs[:] = np.frombuffer(block[:768].tobytes(), dtype="<i2").reshape(24, 16)
            is_i4x4 = int(block[768])
            modes = np.frombuffer(block[769:785].tobytes(), dtype=np.uint8)
            chroma_mode = int(block[785])
            nonzero_y = int.from_bytes(block[788:792].tobytes(), "little")
            nonzero_uv = int.from_bytes(block[792:796].tobytes(), "little")
            skip = int(block[797])
            segment = int(block[798])
            if skip not in (0, 1) or segment > 3:
                raise ValueError("WebP pre-reconstruction metadata contains an invalid skip or segment value")
            features[token, 0] = x / 15.0
            features[token, 1] = y / 15.0
            y_coeff = coeffs[:16].reshape(16, 16)
            u_coeff = coeffs[16:20].reshape(4, 16)
            v_coeff = coeffs[20:24].reshape(4, 16)
            features[token, 2:10] = y_coeff[:, WEBP_LOW_FREQUENCY_POSITIONS].mean(axis=0)
            features[token, 10:18] = u_coeff[:, WEBP_LOW_FREQUENCY_POSITIONS].mean(axis=0)
            features[token, 18:26] = v_coeff[:, WEBP_LOW_FREQUENCY_POSITIONS].mean(axis=0)
            features[token, 26:34] = quant[segment]
            mode_ids = WEBP_MODE_MAP[modes if is_i4x4 else modes[:1]]
            features[token, 34:42] = np.bincount(mode_ids, minlength=8)[:8]
            features[token, 42] = WEBP_CHROMA_MODE_MAP[min(chroma_mode, 3)]
            features[token, 43] = is_i4x4
            features[token, 44] = skip
            features[token, 45] = segment
            features[token, 46] = np.count_nonzero(y_coeff) / y_coeff.size
            features[token, 47] = (np.count_nonzero(u_coeff) + np.count_nonzero(v_coeff)) / (u_coeff.size + v_coeff.size)
            features[token, 50:53] = 1.0
            features[token, 53:61] = 1.0
            features[token, 61] = 1.0
            if nonzero_y == 0 and np.count_nonzero(y_coeff) != 0:
                raise ValueError("WebP luma nonzero metadata conflicts with parsed coefficients")
            if nonzero_uv == 0 and np.count_nonzero(u_coeff) + np.count_nonzero(v_coeff) != 0:
                raise ValueError("WebP chroma nonzero metadata conflicts with parsed coefficients")
    return features
