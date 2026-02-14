"""
Meshy mesh provider implementation.

Uses the Meshy REST API directly via requests.
API key: set MESHY_API_KEY env var or pass via ProviderConfig.
"""
import os
import time
import base64
import logging
import requests

from mesh_provider import (
    MeshProvider, ProviderConfig, GenerationResult, MeshFormat,
    ProviderError, ProviderTimeoutError, ProviderAPIError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)

MESHY_BASE = "https://api.meshy.ai/openapi"
MESHY_TEXT_TO_3D_URL = f"{MESHY_BASE}/v2/text-to-3d"
MESHY_IMAGE_TO_3D_URL = f"{MESHY_BASE}/v1/image-to-3d"


class MeshyProvider(MeshProvider):
    """Mesh provider backed by the Meshy cloud API."""

    @property
    def name(self) -> str:
        return "meshy"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def text_to_mesh(self, prompt, output_path, negative_prompt=""):
        # Preview mode only — geometry without textures is sufficient for
        # flat-pack decomposition. Saves credits and generation time.
        payload = {
            "mode": "preview",
            "prompt": prompt,
            "should_remesh": True,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        logger.info("Submitting text-to-3D task: %s", prompt)
        resp = requests.post(
            MESHY_TEXT_TO_3D_URL, json=payload, headers=self._headers()
        )
        self._check_response(resp)
        task_id = resp.json()["result"]
        logger.info("Task submitted: %s", task_id)

        self._poll_task(f"{MESHY_TEXT_TO_3D_URL}/{task_id}")

        task_data = requests.get(
            f"{MESHY_TEXT_TO_3D_URL}/{task_id}", headers=self._headers()
        ).json()

        mesh_url = self._get_model_url(task_data)
        self._download_file(mesh_url, output_path)

        logger.info("Mesh saved to: %s", output_path)
        return GenerationResult(
            mesh_path=output_path,
            task_id=task_id,
            provider_name=self.name,
            format=MeshFormat.GLB,
        )

    def image_to_mesh(self, image_path, output_path):
        if not os.path.isfile(image_path):
            raise ProviderError(f"Image file not found: {image_path}")

        # Convert local image to base64 data URI
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")
        data_uri = f"data:image/{mime};base64,{image_data}"

        payload = {
            "image_url": data_uri,
            "should_remesh": True,
        }

        logger.info("Submitting image-to-3D task: %s", image_path)
        resp = requests.post(
            MESHY_IMAGE_TO_3D_URL, json=payload, headers=self._headers()
        )
        self._check_response(resp)
        task_id = resp.json()["result"]
        logger.info("Task submitted: %s", task_id)

        self._poll_task(f"{MESHY_IMAGE_TO_3D_URL}/{task_id}")

        task_data = requests.get(
            f"{MESHY_IMAGE_TO_3D_URL}/{task_id}", headers=self._headers()
        ).json()

        mesh_url = self._get_model_url(task_data)
        self._download_file(mesh_url, output_path)

        logger.info("Mesh saved to: %s", output_path)
        return GenerationResult(
            mesh_path=output_path,
            task_id=task_id,
            provider_name=self.name,
            format=MeshFormat.GLB,
        )

    def _poll_task(self, task_url):
        """Poll task status until completion or timeout."""
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > self.config.timeout_seconds:
                raise ProviderTimeoutError(
                    f"Task timed out after {self.config.timeout_seconds}s"
                )

            resp = requests.get(task_url, headers=self._headers())
            self._check_response(resp)
            data = resp.json()
            status = data.get("status", "")
            progress = data.get("progress", 0)
            logger.info(
                "Task %s: %s (%d%%) [%.0fs elapsed]",
                data.get("id", "?"), status, progress, elapsed,
            )

            if status == "SUCCEEDED":
                return
            if status in ("FAILED", "CANCELED"):
                raise ProviderAPIError(f"Task {status}: {data}")

            time.sleep(self.config.poll_interval_seconds)

    def _check_response(self, resp):
        """Check HTTP response for errors."""
        if resp.status_code == 429:
            raise ProviderRateLimitError("Meshy rate limit exceeded")
        if resp.status_code in (401, 403):
            raise ProviderAPIError(
                f"Meshy authentication failed ({resp.status_code}). "
                "Check your MESHY_API_KEY."
            )
        if resp.status_code >= 400:
            raise ProviderAPIError(
                f"Meshy API error {resp.status_code}: {resp.text}"
            )

    def _get_model_url(self, task_data):
        """Extract mesh download URL from task response."""
        model_urls = task_data.get("model_urls", {})
        # Prefer GLB, fall back to OBJ
        for fmt in ("glb", "obj", "fbx"):
            url = model_urls.get(fmt)
            if url:
                return url
        raise ProviderAPIError(
            f"No downloadable model URL in response: {model_urls}"
        )

    def _download_file(self, url, output_path):
        """Download a file from URL to local path."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
