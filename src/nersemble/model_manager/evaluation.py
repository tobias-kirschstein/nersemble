from dataclasses import dataclass
from typing import Dict

from elias.config import Config


@dataclass
class NVSEvaluationMetrics(Config):
    psnr: float
    ssim: float
    lpips: float
    mse: float
    jod: float


@dataclass
class NVSEvaluationMetricsBundle(Config):
    regular: NVSEvaluationMetrics
    masked: NVSEvaluationMetrics


@dataclass
class NVSEvaluationResult(Config):
    mean: NVSEvaluationMetricsBundle
    per_cam: Dict[str, NVSEvaluationMetricsBundle]
