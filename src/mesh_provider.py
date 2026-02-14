"""
Abstract mesh provider interface for cloud 3D generation APIs.

Providers convert text prompts or images to downloadable 3D mesh files.
Both Tripo3D and Meshy implement this interface.
"""
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MeshFormat(Enum):
    """Supported 3D mesh output formats."""
    GLB = "glb"
    OBJ = "obj"
    STL = "stl"
    FBX = "fbx"


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderTimeoutError(ProviderError):
    """Task took too long to complete."""
    pass


class ProviderAPIError(ProviderError):
    """API returned an error."""
    pass


class ProviderRateLimitError(ProviderError):
    """API rate limit hit."""
    pass


@dataclass
class GenerationResult:
    """Result from a mesh generation task."""
    mesh_path: str              # Path to downloaded mesh file
    task_id: str                # Provider's task ID
    provider_name: str          # "tripo" or "meshy"
    format: MeshFormat          # Output format
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for a mesh provider."""
    api_key: str
    output_format: MeshFormat = MeshFormat.GLB
    timeout_seconds: float = 300.0      # 5 minute default
    poll_interval_seconds: float = 5.0
    output_dir: str = "generated_meshes"


class MeshProvider(ABC):
    """Abstract base class for cloud 3D mesh generation providers.

    Implementations must handle:
    - API authentication
    - Async task submission and polling
    - Downloading the resulting mesh file
    - Error handling (timeouts, rate limits, failures)
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'tripo', 'meshy')."""
        ...

    @abstractmethod
    def text_to_mesh(self, prompt: str, output_path: str,
                     negative_prompt: str = "") -> GenerationResult:
        """Generate a 3D mesh from a text prompt.

        Args:
            prompt: Text description of the desired 3D object.
            output_path: File path where the mesh should be saved.
            negative_prompt: Features to avoid in generation.

        Returns:
            GenerationResult with the path to the downloaded mesh.

        Raises:
            ProviderTimeoutError: If generation exceeds timeout.
            ProviderAPIError: If the API returns an error.
            ProviderRateLimitError: If rate limited.
        """
        ...

    @abstractmethod
    def image_to_mesh(self, image_path: str, output_path: str) -> GenerationResult:
        """Generate a 3D mesh from an input image.

        Args:
            image_path: Path to the input image (JPG, PNG).
            output_path: File path where the mesh should be saved.

        Returns:
            GenerationResult with the path to the downloaded mesh.

        Raises:
            ProviderTimeoutError: If generation exceeds timeout.
            ProviderAPIError: If the API returns an error.
            ProviderRateLimitError: If rate limited.
        """
        ...
