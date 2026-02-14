"""
Tripo3D mesh provider implementation.

Uses the official tripo3d Python SDK (pip install tripo3d).
API key: set TRIPO_API_KEY env var or pass via ProviderConfig.
"""
import asyncio
import os
import logging

from mesh_provider import (
    MeshProvider, ProviderConfig, GenerationResult, MeshFormat,
    ProviderError, ProviderTimeoutError, ProviderAPIError,
)

logger = logging.getLogger(__name__)


class TripoProvider(MeshProvider):
    """Mesh provider backed by the Tripo3D cloud API."""

    @property
    def name(self) -> str:
        return "tripo"

    def text_to_mesh(self, prompt, output_path, negative_prompt=""):
        return asyncio.run(
            self._text_to_mesh_async(prompt, output_path, negative_prompt)
        )

    def image_to_mesh(self, image_path, output_path):
        if not os.path.isfile(image_path):
            raise ProviderError(f"Image file not found: {image_path}")
        return asyncio.run(self._image_to_mesh_async(image_path, output_path))

    async def _text_to_mesh_async(self, prompt, output_path, negative_prompt):
        from tripo3d import TripoClient

        async with TripoClient(api_key=self.config.api_key) as client:
            logger.info("Submitting text-to-3D task: %s", prompt)
            try:
                task_id = await client.text_to_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "",
                )
            except Exception as e:
                raise ProviderAPIError(f"Tripo text_to_model failed: {e}") from e

            logger.info("Task submitted: %s, waiting for completion...", task_id)
            try:
                task = await asyncio.wait_for(
                    client.wait_for_task(task_id, verbose=True),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise ProviderTimeoutError(
                    f"Task {task_id} timed out after {self.config.timeout_seconds}s"
                )
            except Exception as e:
                raise ProviderAPIError(f"Tripo wait_for_task failed: {e}") from e

            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            try:
                downloaded = await client.download_task_models(task, output_dir)
            except Exception as e:
                raise ProviderAPIError(f"Tripo download failed: {e}") from e

            # Find the downloaded file and rename to output_path
            mesh_path = self._find_and_rename(downloaded, output_path)

            logger.info("Mesh saved to: %s", mesh_path)
            return GenerationResult(
                mesh_path=mesh_path,
                task_id=str(task_id),
                provider_name=self.name,
                format=self.config.output_format,
            )

    async def _image_to_mesh_async(self, image_path, output_path):
        from tripo3d import TripoClient

        async with TripoClient(api_key=self.config.api_key) as client:
            logger.info("Submitting image-to-3D task: %s", image_path)
            try:
                task_id = await client.image_to_model(image=image_path)
            except Exception as e:
                raise ProviderAPIError(f"Tripo image_to_model failed: {e}") from e

            logger.info("Task submitted: %s, waiting for completion...", task_id)
            try:
                task = await asyncio.wait_for(
                    client.wait_for_task(task_id, verbose=True),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise ProviderTimeoutError(
                    f"Task {task_id} timed out after {self.config.timeout_seconds}s"
                )
            except Exception as e:
                raise ProviderAPIError(f"Tripo wait_for_task failed: {e}") from e

            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            try:
                downloaded = await client.download_task_models(task, output_dir)
            except Exception as e:
                raise ProviderAPIError(f"Tripo download failed: {e}") from e

            mesh_path = self._find_and_rename(downloaded, output_path)

            logger.info("Mesh saved to: %s", mesh_path)
            return GenerationResult(
                mesh_path=mesh_path,
                task_id=str(task_id),
                provider_name=self.name,
                format=self.config.output_format,
            )

    def _find_and_rename(self, downloaded, output_path):
        """Find the first downloaded mesh file and rename to output_path."""
        # downloaded is a dict mapping model_type -> file_path
        for model_type, file_path in downloaded.items():
            if file_path and os.path.isfile(file_path):
                if file_path != output_path:
                    os.rename(file_path, output_path)
                return output_path
        raise ProviderAPIError("No mesh file was downloaded from Tripo")
