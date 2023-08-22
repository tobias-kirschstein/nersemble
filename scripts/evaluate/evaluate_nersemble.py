import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional, Union

import numpy as np
import pyfvvdp
import torch
import tyro
from nerfacc import OccGridEstimator
from nerfstudio.data.utils.dataloaders import EvalDataloader
from tqdm import tqdm

from nersemble.constants import SERIALS, EVALUATION_CAM_IDS
from nersemble.model_manager.evaluation import NVSEvaluationResult, NVSEvaluationMetricsBundle, NVSEvaluationMetrics
from nersemble.model_manager.nersemble import NeRSembleModelFolder
from nersemble.util.connected_components import filter_occupancy_grid
from nersemble.util.setup import nersemble_eval_setup


def perform_alpha_blending(image: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
    assert image.dtype == np.uint8
    assert alpha_map.dtype == np.uint8
    assert image.shape[:2] == alpha_map.shape[:2]

    image = image / 255.
    alpha_map = alpha_map / 255.
    image = alpha_map * image + (1 - alpha_map) * np.ones_like(image)

    image = image * 255.
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    return image


def main(run_name: str,
         checkpoint: Optional[Union[str, int]] = None,
         /,
         n_rays_eval: int = 2 ** 13,
         max_eval_timesteps: int = 15,
         skip_timesteps: Optional[int] = None,
         use_occupancy_grid_filtering: bool = True,
         occupancy_grid_filtering_threshold: float = 0.05,
         occupancy_grid_filtering_sigma_erosion: int = 7,
         ):
    jod_evaluator = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')

    # Setup Model Manager and load key configs from train run
    model_manager = NeRSembleModelFolder().open_run(run_name)

    rgb_channel_name = 'rgb'

    # Setup pipeline and load model
    config_path = Path(model_manager.get_config_path())

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------

    _, pipeline, checkpoint_path, _ = nersemble_eval_setup(config_path,
                                                           model_manager.get_checkpoint_folder(),
                                                           max_eval_timesteps=max_eval_timesteps,
                                                           eval_num_rays_per_chunk=n_rays_eval,
                                                           eval_num_images_to_sample_from=36)

    if use_occupancy_grid_filtering:
        # Ensure that eval occupancy grid only contains one large blob and no floaters
        occupancy_grid: OccGridEstimator = pipeline.model.occupancy_grid
        filter_occupancy_grid(occupancy_grid,
                              threshold=occupancy_grid_filtering_threshold,
                              sigma_erosion=occupancy_grid_filtering_sigma_erosion)

    if checkpoint is None:
        checkpoint_path_matches = re.match("step-(\d+)\.ckpt", Path(checkpoint_path).name)
        checkpoint = int(checkpoint_path_matches.group(1))

    name_parts = []

    if max_eval_timesteps > 0:
        name_parts.append(f"max_eval_timesteps_{max_eval_timesteps}")

    if skip_timesteps is not None and skip_timesteps > 1:
        name_parts.append(f"skip_timesteps_{skip_timesteps}")

    if not use_occupancy_grid_filtering:
        name_parts.append(f"no-occupancy-grid-filtering")

    if name_parts:
        name = '_'.join(name_parts)
    else:
        name = None

    # Setup cameras for evaluation views
    eval_dataloader: EvalDataloader = pipeline.datamanager.eval_dataloader
    cameras = eval_dataloader.cameras
    assert cameras.times is not None or cameras.size == 4, "Expected 4 validation cameras to be available"
    assert cameras.times is None or max_eval_timesteps == -1 or cameras.size == 4 * max_eval_timesteps, "Expected 4x n_timesteps cameras to be available"

    psnrs = []
    mses = []
    lpipses = []
    ssims = []

    masked_psnrs = []
    masked_mses = []
    masked_lpipses = []
    masked_ssims = []

    psnr_key = 'psnr'
    mse_key = 'mse'
    lpips_key = 'lpips'
    ssim_key = 'ssim'

    masked_psnr_key = 'psnr_masked'
    masked_mse_key = 'mse_masked'
    masked_lpips_key = 'lpips_masked'
    masked_ssim_key = 'ssim_masked'

    evaluated_timesteps = []
    evaluated_cam_ids = []

    # ----------------------------------------------------------
    # Generate predictions
    # ----------------------------------------------------------

    images_predicted = defaultdict(list)
    images_predicted_masked = defaultdict(list)
    images_gt = defaultdict(list)
    images_gt_masked = defaultdict(list)

    for camera_ray_bundle, batch in tqdm(pipeline.datamanager.fixed_indices_eval_dataloader,
                                         desc="Generating predictions"):

        time = camera_ray_bundle.times.flatten()[0].item()
        timestep = pipeline.datamanager.dataparser.config.time_to_original_timestep(time)

        if skip_timesteps is None or timestep % skip_timesteps == 0:
            # Only evaluate every skip_timesteps-th timestep
            cam_id = int(batch["cam_ids"])
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                metrics_dict, images_dict = pipeline.model.get_image_metrics_and_images(outputs, batch)

            image = outputs[rgb_channel_name].cpu().numpy()
            model_manager.save_evaluation_img(cam_id, image, checkpoint=checkpoint, timestep=timestep,
                                              max_eval_timesteps=max_eval_timesteps,
                                              skip_timesteps=skip_timesteps,
                                              use_occupancy_grid_filtering=use_occupancy_grid_filtering)

            # Collect images for JOD metric

            image_gt = (batch['image'] * 255).cpu().numpy().astype(np.uint8)
            image_predicted = (image * 255).astype(np.uint8)

            images_gt[cam_id].append(image_gt)
            images_predicted[cam_id].append(image_predicted)

            if "alpha_map" in batch:
                alpha_map = batch['alpha_map']

                image_predicted_masked = perform_alpha_blending(image_predicted, alpha_map)
                image_gt_masked = perform_alpha_blending(image_gt, alpha_map)

                images_predicted_masked[cam_id].append(image_predicted_masked)
                images_gt_masked[cam_id].append(image_gt_masked)

            # Collect image-level metrics
            psnrs.append(metrics_dict[psnr_key])
            mses.append(metrics_dict[mse_key])
            lpipses.append(metrics_dict[lpips_key])
            ssims.append(metrics_dict[ssim_key])

            if masked_psnr_key is not None and masked_psnr_key in metrics_dict:
                masked_psnrs.append(metrics_dict[masked_psnr_key])
            if masked_mse_key is not None and masked_mse_key in metrics_dict:
                masked_mses.append(metrics_dict[masked_mse_key])
            if masked_lpips_key is not None and masked_lpips_key in metrics_dict:
                masked_lpipses.append(metrics_dict[masked_lpips_key])
            if masked_ssim_key is not None and masked_ssim_key in metrics_dict:
                masked_ssims.append(metrics_dict[masked_ssim_key])

            print(metrics_dict)

            evaluated_timesteps.append(timestep)
            evaluated_cam_ids.append(cam_id)

    # ----------------------------------------------------------
    # Collect metrics
    # ----------------------------------------------------------

    psnr = mean(psnrs)
    mse = mean(mses)
    lpips = mean(lpipses)
    ssim = mean(ssims)

    masked_psnr = mean(masked_psnrs) if masked_psnrs else None
    masked_mse = mean(masked_mses) if masked_mses else None
    masked_lpips = mean(masked_lpipses) if masked_lpipses else None
    masked_ssim = mean(masked_ssims) if masked_ssims else None

    # ----------------------------------------------------------
    # JOD metric
    # ----------------------------------------------------------
    dataparser_config = pipeline.datamanager.dataparser.config
    evaluation_fps = 73
    evaluation_fps /= dataparser_config.skip_timesteps
    if skip_timesteps is not None and skip_timesteps > 1:
        evaluation_fps /= skip_timesteps
    elif max_eval_timesteps > 0:
        evaluation_fps /= (dataparser_config.n_timesteps / max_eval_timesteps)

    jod_per_cam = []
    masked_jod_per_cam = []
    n_evaluation_cams = len(images_predicted)
    for cam_id in range(n_evaluation_cams):
        cam_images_predicted = np.stack(images_predicted[cam_id])  # [T, H, W, C]
        cam_images_gt = np.stack(images_gt[cam_id])  # [T, H, W, C]

        jod, _ = jod_evaluator.predict(cam_images_predicted, cam_images_gt, dim_order="FHWC",
                                       frames_per_second=max(4.1, evaluation_fps))
        jod_per_cam.append(jod.item())

        if cam_id in images_predicted_masked:
            cam_images_predicted_masked = np.stack(images_predicted_masked[cam_id])  # [T, H, W, C]
            cam_images_gt_masked = np.stack(images_gt_masked[cam_id])  # [T, H, W, C]

            masked_jod, _ = jod_evaluator.predict(cam_images_predicted_masked, cam_images_gt_masked,
                                                  dim_order="FHWC",
                                                  frames_per_second=max(4.1, evaluation_fps))

            masked_jod_per_cam.append(masked_jod.item())

    jod = mean(jod_per_cam)
    if masked_jod_per_cam:
        masked_jod = mean(masked_jod_per_cam)
    else:
        masked_jod = None

    print("======================================================")
    print(f"Evaluation Result for {run_name} checkpoint {checkpoint}")
    print("======================================================")
    print(f"PSNR: {psnr: 0.2f}")
    print(f"SSIM: {ssim: 0.3f}")
    print(f"LPIPS: {lpips: 0.3f}")
    print(f"MSE: {mse: 0.4f}")
    print(f"JOD: {jod: 0.3f}")
    if masked_psnr is not None:
        print(f"PSNR (masked): {masked_psnr: 0.2f}")
    if masked_ssim is not None:
        print(f"SSIM (masked): {masked_ssim: 0.3f}")
    if masked_lpips is not None:
        print(f"LPIPS (masked): {masked_lpips: 0.3f}")
    if masked_mse is not None:
        print(f"MSE (masked): {masked_mse: 0.4f}")
    if masked_jod is not None:
        print(f"JOD (masked): {masked_jod: 0.3f}")

    # ----------------------------------------------------------
    # Store evaluation result
    # ----------------------------------------------------------

    per_cam_metrics = dict()
    for cam_id in range(n_evaluation_cams):
        per_cam_psnr = mean([metric for j, metric in enumerate(psnrs) if evaluated_cam_ids[j] == cam_id])
        per_cam_ssim = mean([metric for j, metric in enumerate(ssims) if evaluated_cam_ids[j] == cam_id])
        per_cam_lpips = mean([metric for j, metric in enumerate(lpipses) if evaluated_cam_ids[j] == cam_id])
        per_cam_mse = mean([metric for j, metric in enumerate(mses) if evaluated_cam_ids[j] == cam_id])

        per_cam_masked_psnr = None
        per_cam_masked_ssim = None
        per_cam_masked_lpips = None
        per_cam_masked_mse = None

        if masked_psnrs is not None:
            per_cam_masked_psnr = mean(
                [metric for j, metric in enumerate(masked_psnrs) if evaluated_cam_ids[j] == cam_id])
        if masked_ssims is not None:
            per_cam_masked_ssim = mean(
                [metric for j, metric in enumerate(masked_ssims) if evaluated_cam_ids[j] == cam_id])
        if masked_lpipses is not None:
            per_cam_masked_lpips = mean(
                [metric for j, metric in enumerate(masked_lpipses) if evaluated_cam_ids[j] == cam_id])
        if masked_mses is not None:
            per_cam_masked_mse = mean(
                [metric for j, metric in enumerate(masked_mses) if evaluated_cam_ids[j] == cam_id])

        serial = SERIALS[EVALUATION_CAM_IDS[cam_id]]
        per_cam_metrics[serial] = NVSEvaluationMetricsBundle(
            regular=NVSEvaluationMetrics(per_cam_psnr,
                                         per_cam_ssim,
                                         per_cam_lpips,
                                         per_cam_mse,
                                         jod_per_cam[cam_id]),
            masked=NVSEvaluationMetrics(per_cam_masked_psnr,
                                        per_cam_masked_ssim,
                                        per_cam_masked_lpips,
                                        per_cam_masked_mse,
                                        masked_jod_per_cam[cam_id])
        )

    # Total results: averaged over all evaluation frames
    evaluation_result = NVSEvaluationResult(
        mean=NVSEvaluationMetricsBundle(
            regular=NVSEvaluationMetrics(psnr, ssim, lpips, mse, jod),
            masked=NVSEvaluationMetrics(masked_psnr, masked_ssim, masked_lpips, masked_mse, masked_jod),
        ),
        per_cam=per_cam_metrics
    )

    model_manager.save_evaluation_result(evaluation_result,
                                         checkpoint=checkpoint,
                                         max_eval_timesteps=max_eval_timesteps,
                                         skip_timesteps=skip_timesteps,
                                         use_occupancy_grid_filtering=use_occupancy_grid_filtering)


if __name__ == '__main__':
    tyro.cli(main)
