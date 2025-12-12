import os
import sys
import types
import unittest
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, json_data, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class TestApiUtils(unittest.TestCase):
    def test_load_env_api_key_text_priority(self):
        # Avoid loading any real .env
        fake_dotenv = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            from paper2slides.utils.api_utils import load_env_api_key

            with patch.dict(os.environ, {"OPENAI_API_KEY": " openai "}, clear=True):
                self.assertEqual(load_env_api_key("text"), "openai")

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "openai",
                    "RUNWAY_API_KEY": "runway",
                    "GEMINI_TEXT_KEY": "gemini",
                    "RAG_LLM_API_KEY": "rag",
                },
                clear=True,
            ):
                self.assertEqual(load_env_api_key("text"), "rag")

    def test_load_env_api_key_image_priority(self):
        fake_dotenv = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            from paper2slides.utils.api_utils import load_env_api_key

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "openai",
                    "RUNWAY_API_KEY": "runway",
                    "GEMINI_IMAGE_API_KEY": "gemini_img",
                    "IMAGE_GEN_API_KEY": "img_key",
                },
                clear=True,
            ):
                self.assertEqual(load_env_api_key("image"), "img_key")

    def test_get_api_base_url_priority(self):
        from paper2slides.utils.api_utils import get_api_base_url

        with patch.dict(
            os.environ,
            {
                "RUNWAY_API_BASE": "https://runway.example/v1",
                "OPENAI_BASE_URL": "https://openai-proxy.example/v1",
                "RAG_LLM_BASE_URL": "https://rag.example/v1",
                "IMAGE_GEN_BASE_URL": "https://img.example/v1",
            },
            clear=True,
        ):
            self.assertEqual(get_api_base_url("text"), "https://rag.example/v1")
            self.assertEqual(get_api_base_url("image"), "https://img.example/v1")

        with patch.dict(
            os.environ,
            {
                "RUNWAY_API_BASE": "https://runway.example/v1",
                "OPENAI_BASE_URL": "https://openai-proxy.example/v1",
                "RAG_LLM_BASE_URL": "https://rag.example/v1",
            },
            clear=True,
        ):
            # image falls back to text URL if IMAGE_GEN_BASE_URL not set
            self.assertEqual(get_api_base_url("image"), "https://rag.example/v1")

    def test_custom_http_chat_completions_create_builds_request(self):
        from paper2slides.utils.api_utils import CustomHTTPClient

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch("paper2slides.utils.api_utils.requests.post", side_effect=fake_post):
            client = CustomHTTPClient(api_key="k", base_url="https://runway.example/v1/")
            resp = client.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                extra_body={"should": "be removed"},
            )

        self.assertIn("/chat/completions", captured["url"])
        self.assertIn("api-version=2024-12-01-preview", captured["url"])
        self.assertEqual(captured["headers"]["api-key"], "k")
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")
        self.assertEqual(captured["json"]["model"], "m")
        self.assertEqual(captured["json"]["messages"][0]["content"], "hi")
        self.assertEqual(captured["json"]["temperature"], 0.2)
        self.assertNotIn("extra_body", captured["json"])
        self.assertEqual(resp.choices[0].message.content, "ok")

    def test_custom_http_embeddings_create_builds_request(self):
        from paper2slides.utils.api_utils import CustomHTTPClient

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse({"data": [{"embedding": [0.1, 0.2]}]})

        with patch("paper2slides.utils.api_utils.requests.post", side_effect=fake_post):
            client = CustomHTTPClient(api_key="k", base_url="https://devops.example/v1")
            resp = client.embeddings.create(model="e", input="hello")

        self.assertIn("/embeddings", captured["url"])
        self.assertIn("api-version=2024-12-01-preview", captured["url"])
        self.assertEqual(captured["headers"]["api-key"], "k")
        self.assertEqual(captured["json"]["model"], "e")
        self.assertEqual(captured["json"]["input"], "hello")
        self.assertEqual(resp.data[0].embedding, [0.1, 0.2])

    def test_get_openai_client_returns_custom_http_for_runway(self):
        from paper2slides.utils.api_utils import CustomHTTPClient, get_openai_client

        with patch.dict(os.environ, {}, clear=True):
            client = get_openai_client(api_key="k", base_url="https://runway.example/v1", key_type="text")

        self.assertIsInstance(client, CustomHTTPClient)

    def test_get_openai_client_returns_openai_for_normal_url(self):
        from paper2slides.utils.api_utils import get_openai_client

        captured = {}

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url

        fake_openai_mod = types.SimpleNamespace(OpenAI=FakeOpenAI)

        with patch.dict(sys.modules, {"openai": fake_openai_mod}):
            client = get_openai_client(api_key="k", base_url="https://api.openai.com/v1", key_type="text")

        self.assertIsInstance(client, FakeOpenAI)
        self.assertEqual(captured["api_key"], "k")
        self.assertEqual(captured["base_url"], "https://api.openai.com/v1")

    def test_get_openai_client_raises_if_no_key(self):
        from paper2slides.utils.api_utils import get_openai_client

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                get_openai_client(api_key=None, base_url="https://api.openai.com/v1", key_type="text")


if __name__ == "__main__":
    unittest.main()


