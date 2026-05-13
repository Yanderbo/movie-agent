# -*- coding: utf-8 -*-
"""
OCR 识别
OCR 功能已与画面摘要合并到 vision.py 中，通过一次 Gemini Vision API 调用
同时完成 OCR 和画面描述，以减少 API 调用次数。

如需单独使用 OCR，请调用 vision.analyze_keyframes() 并取返回值的第一个元素。
"""
from pipeline.vision import analyze_keyframes  # noqa: F401
