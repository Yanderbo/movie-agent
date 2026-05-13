# -*- coding: utf-8 -*-
"""
LLM 客户端封装
统一通过中转 API（OpenAI 兼容格式）调用 Gemini。
支持：纯文本、图片输入、音频输入、视频输入。
"""
import os
import re
import json
import time
import base64
import requests
from pathlib import Path

from utils.logger import get_logger
import config

logger = get_logger("LLMClient")


class LLMClient:
    """
    基于中转 API 的 LLM 客户端。
    API 格式兼容 OpenAI /chat/completions，支持流式响应。
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = None,
        timeout: int = None,
    ):
        self.api_key = api_key or config.LLM_API_KEY
        self.api_base = (api_base or config.LLM_API_BASE).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.base_url = f"{self.api_base}/chat/completions"

        if not self.api_key:
            logger.warning("LLM_API_KEY 未设置，LLM 调用将失败")

    # ─── 编码辅助 ────────────────────────────────────────────

    @staticmethod
    def encode_file_to_base64(file_path: str) -> str:
        """将文件编码为 base64 字符串"""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        """根据扩展名推断 MIME 类型"""
        ext = Path(file_path).suffix.lower()
        mime_map = {
            # 视频
            ".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            # 图片
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".gif": "image/gif",
            # 音频
            ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".flac": "audio/flac", ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }
        return mime_map.get(ext, "application/octet-stream")

    # ─── 请求构造 ────────────────────────────────────────────

    def _build_messages_text(self, prompt: str, system_prompt: str = None) -> list:
        """构造纯文本消息"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_messages_with_media(
        self, prompt: str, media_path: str, system_prompt: str = None
    ) -> list:
        """构造带媒体文件（图片/视频/音频）的消息"""
        mime_type = self.get_mime_type(media_path)
        b64_data = self.encode_file_to_base64(media_path)
        data_url = f"data:{mime_type};base64,{b64_data}"

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": data_url, "mime_type": mime_type},
            },
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _build_messages_with_images(
        self, prompt: str, image_paths: list[str], system_prompt: str = None
    ) -> list:
        """构造带多张图片的消息"""
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            mime_type = self.get_mime_type(img_path)
            b64_data = self.encode_file_to_base64(img_path)
            data_url = f"data:{mime_type};base64,{b64_data}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "mime_type": mime_type},
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    # ─── API 请求 ─────────────────────────────────────────────

    def _request(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = None,
        response_format: str = None,
    ) -> dict:
        """
        发送请求到中转 API，流式接收响应。

        Returns:
            {"success": bool, "content": str, "error": str, "response_time": float}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        result = {"success": False, "content": "", "error": "", "response_time": 0.0}
        start_time = time.time()

        try:
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.error(f"API 请求失败: {result['error']}")
                return result

            result["success"] = True
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else line
                if text.startswith("data:"):
                    text = text[len("data:"):].strip()
                    if text == "[DONE]":
                        break
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                        if "choices" in obj and obj["choices"]:
                            choice = obj["choices"][0]
                            if "delta" in choice:
                                chunk = choice["delta"].get("content", "") or ""
                            elif "text" in choice:
                                chunk = choice.get("text", "")
                            else:
                                chunk = ""
                            if chunk:
                                result["content"] += chunk
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            result["error"] = f"请求超时 ({self.timeout}秒)"
            logger.error(result["error"])
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"API 请求异常: {e}")

        result["response_time"] = time.time() - start_time
        return result

    # ─── 重试封装 ─────────────────────────────────────────────

    def _request_with_retry(self, messages, max_retries=3, **kwargs) -> dict:
        """带指数退避重试的请求"""
        last_result = None
        for attempt in range(max_retries):
            result = self._request(messages, **kwargs)
            if result["success"] and result["content"].strip():
                return result
            last_result = result
            wait = 2 ** attempt
            logger.warning(
                f"请求失败 (尝试 {attempt+1}/{max_retries})，{wait}秒后重试: "
                f"{result.get('error', '空响应')}"
            )
            time.sleep(wait)
        return last_result or {"success": False, "content": "", "error": "全部重试失败"}

    # ─── 公开接口 ─────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        response_format: str = None,
    ) -> str:
        """纯文本对话，返回文本内容"""
        messages = self._build_messages_text(prompt, system_prompt)
        result = self._request_with_retry(
            messages, temperature=temperature,
            max_tokens=max_tokens, response_format=response_format,
        )
        if not result["success"]:
            raise RuntimeError(f"LLM 调用失败: {result['error']}")
        return result["content"]

    def chat_with_media(
        self,
        prompt: str,
        media_path: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        response_format: str = None,
    ) -> str:
        """带媒体文件的对话（图片/视频/音频），返回文本内容"""
        if not Path(media_path).exists():
            raise FileNotFoundError(f"媒体文件不存在: {media_path}")
        messages = self._build_messages_with_media(prompt, media_path, system_prompt)
        result = self._request_with_retry(
            messages, temperature=temperature,
            max_tokens=max_tokens, response_format=response_format,
        )
        if not result["success"]:
            raise RuntimeError(f"LLM 调用失败: {result['error']}")
        return result["content"]

    def chat_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        response_format: str = None,
    ) -> str:
        """带多张图片的对话，返回文本内容"""
        for p in image_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"图片文件不存在: {p}")
        messages = self._build_messages_with_images(
            prompt, image_paths, system_prompt
        )
        result = self._request_with_retry(
            messages, temperature=temperature,
            max_tokens=max_tokens, response_format=response_format,
        )
        if not result["success"]:
            raise RuntimeError(f"LLM 调用失败: {result['error']}")
        return result["content"]

    def chat_json(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = None,
    ) -> dict | list:
        """纯文本对话，返回解析后的 JSON 对象"""
        content = self.chat(
            prompt, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )
        return self.parse_json(content)

    def chat_with_media_json(
        self,
        prompt: str,
        media_path: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = None,
    ) -> dict | list:
        """带媒体文件的对话，返回解析后的 JSON 对象"""
        content = self.chat_with_media(
            prompt, media_path, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )
        return self.parse_json(content)

    # ─── JSON 解析 ────────────────────────────────────────────

    @staticmethod
    def parse_json(content: str) -> dict | list | None:
        """从 LLM 响应中解析 JSON，支持 markdown 代码块包裹"""
        if not content:
            return None

        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'(\[[\s\S]*\])',
            r'(\{[\s\S]*\})',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            logger.error(f"无法解析 JSON: {content[:200]}...")
            return None


# ─── 全局单例 ─────────────────────────────────────────────────

_client: LLMClient = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
