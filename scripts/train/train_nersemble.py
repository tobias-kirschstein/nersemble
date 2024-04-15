from pathlib import Path
from typing import Literal, Optional

import os
import torch
import tyro
from elias.util.random import make_deterministic
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nersemble.data_manager.multi_view_data import NeRSembleDataManager
from nersemble.model_manager.nersemble import NeRSembleModelFolder
from nersemble.nerfstudio.config.nersemble_trainer_config import NeRSembleTrainerConfig
from nersemble.nerfstudio.datamanager.nersemble_datamanager import NeRSembleVanillaDataManagerConfig
from nersemble.nerfstudio.dataparser.nersemble_dataparser import NeRSembleDataParserConfig
from nersemble.nerfstudio.engine.nersemble_trainer import NeRSembleTrainer
from nersemble.nerfstudio.engine.step_lr_scheduler import StepLRSchedulerConfig
from nersemble.nerfstudio.field_components.deformation_field import SE3DeformationFieldConfig
from nersemble.nerfstudio.field_components.hash_ensemble import HashEnsembleConfig, TCNNHashEncodingConfig
from nersemble.nerfstudio.models.nersemble_instant_ngp import NeRSembleNGPModelConfig

from torchinfo import summary

from nersemble.util.setup import try_load_config

# [left, forward, down], [right, backward, up]
# [[-2.5, -2, -2.5], [2.5, 3, 2]]]
# SCENE_BOXES = {
#     18: [[-1.8, -1.8, -2.5], [1.8, 1.8, 2]],
#     30: [[-2.5, -0.5, -2.5], [2.2, 3.2, 2]],
#     38: [[-1.8, -0.5, -2.5], [2.2, 3.2, 2]],
#     85: [[-2, -1.3, -2.5], [2.2, 2.2, 2]],
#     97: [[-2.2, -1.8, -2.5], [2.2, 3.2, 2]],
#     124: [[-2.2, -1.5, -2.5], [2.2, 2.5, 2]],
#     175: [[-2.3, -0.7, -2.5], [2, 3.2, 2]],
# }

# Due to the newer align_poses(), the scene boxes have to be adapted
SCENE_BOXES = {
    18: [[-1.8, -2.3, -2.5], [1.8, 1.3, 2]],
    30: [[-2.5, -1.8, -2.5], [2.2, 1.8, 2]],
    38: [[-1.8, -1.5, -2.5], [2.2, 2.2, 2]],
    85: [[-2, -1.8, -2.5], [2.2, 1.7, 2]],
    97: [[-2.2, -2.8, -2.5], [2.2, 2.2, 2]],
    124: [[-2.2, -2.5, -2.5], [2.2, 1.5, 2]],
    175: [[-2.3, -2, -2.5], [2, 2, 2]],
}


def main(
        participant_id: int,
        sequence_name: str,
        /,
        name: Optional[str] = None,
        vis: Literal['viewer', 'wandb'] = 'wandb',

        # Sequence
        start_timestep: int = 0,
        n_timesteps: int = -1,
        skip_timesteps: int = 1,
        max_cached_images: int = 10000,  # 10k should be ~200GB RAM

        # Learning rates
        lr_main: float = 5e-3,
        lr_deformation_field: float = 1e-3,
        lr_embeddings: float = 5e-3,

        # Losses
        lambda_alpha_loss: float = 1e-2,
        lambda_near_loss: float = 1e-4,
        lambda_empty_loss: float = 1e-2,
        lambda_depth_loss: float = 1e-4,
        lambda_dist_loss: float = 1e-4,

        # Scheduler
        window_hash_encodings_begin: int = 40000,
        window_hash_encodings_end: int = 80000,
        window_deform_begin: int = 0,
        window_deform_end: int = 20000,

        # Hash Ensemble
        use_hash_ensemble: bool = True,
        n_hash_encodings: int = 32,
        latent_dim_time: int = 32,

        # Deformation Field
        use_deformation_field: bool = True,
        latent_dim_time_deform: int = 128,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,

        # Logging
        steps_per_eval_image: int = 20000,
        steps_per_eval_all_images: int = 50000,

        # Ray Marching
        cone_angle: float = 0,
        alpha_thre: float = 1e-2,
        occ_thre: float = 1e-2,
        n_train_rays: int = 4096,
        grid_levels: int = 1,
        disable_occupancy_grid: bool = False,
        max_n_samples_per_batch: int = -1,

        # View Frustum Culling
        use_view_frustum_culling: bool = True,
        view_frustum_culling: int = 2,

        resume_run: Optional[str] = None,
        resume_checkpoint: Optional[int] = None,

):
    os.environ['WANDB_RUN_GROUP'] = "nersemble"

    seed = 19980801
    make_deterministic(seed)

    model_folder = NeRSembleModelFolder()
    model_manager = model_folder.new_run(name=name)
    run_name = model_manager.get_run_name()

    # TODO: For some reason Instant NGP only really works if we scale the world coordinate system
    scale_factor = 9
    if n_timesteps == -1:
        data_manager = NeRSembleDataManager(participant_id, sequence_name)
        n_timesteps = data_manager.get_n_timesteps()

    if participant_id in SCENE_BOXES:
        scene_box = torch.tensor(SCENE_BOXES[participant_id]) * scale_factor / 9
    else:
        # Default scene box
        # [[left, head front, down], [right, head back, up]]
        scene_box = torch.tensor([[-2.5, -2, -2.5], [2.5, 3, 2]]) * scale_factor / 9

    if resume_run:
        existing_model_manager = NeRSembleModelFolder().open_run(resume_run)
        config = try_load_config(existing_model_manager.get_config_path())
        config.experiment_name = run_name
        config.run_name = run_name
        config.output_dir = Path(model_folder.get_location())

        config.load_dir = Path(existing_model_manager.get_checkpoint_folder())
        config.load_step = resume_checkpoint
    else:
        config = NeRSembleTrainerConfig(
            project_name="nersemble",
            experiment_name=run_name,
            method_name="nersemble",

            output_dir=Path(model_folder.get_location()),
            run_name=run_name,

            steps_per_eval_batch=500,
            steps_per_eval_image=steps_per_eval_image,
            steps_per_eval_all_images=steps_per_eval_all_images,
            steps_per_save=50000,
            max_num_iterations=300001,
            save_only_latest_checkpoint=True,
            mixed_precision=True,
            log_gradients=False,

            pipeline=VanillaPipelineConfig(
                datamanager=NeRSembleVanillaDataManagerConfig(
                    dataparser=NeRSembleDataParserConfig(
                        participant_id=participant_id,
                        sequence_name=sequence_name,
                        start_timestep=start_timestep,
                        n_timesteps=n_timesteps,
                        skip_timesteps=skip_timesteps,
                        scale_factor=scale_factor,
                        scene_box=scene_box,

                    ),
                    train_num_rays_per_batch=n_train_rays,
                    eval_num_rays_per_batch=1024,
                    train_num_images_to_sample_from=24,
                    train_num_times_to_repeat_images=20,
                    eval_num_images_to_sample_from=36,
                    use_cache_compression=False,
                    max_cached_items=max_cached_images,  # 10k should be roughly 200GB
                ),

                model=NeRSembleNGPModelConfig(
                    # Ray Marching
                    render_step_size=0.011 * scale_factor / 9.,  # TODO: Is render_step_size correct?
                    near_plane=0.2 * scale_factor / 9.,
                    far_plane=1e3 * scale_factor / 9.,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,  # TODO: Do lower values help? It seems like the occupancy grid is too aggressive
                    occ_thre=occ_thre,
                    early_stop_eps=0,  # Important, otherwise scene may start exploding
                    background_color="white",
                    grid_levels=grid_levels,
                    # Originally, NeRSemble was trained with a single grid, but larger values also work
                    disable_scene_contraction=True,  # To ensure scene only exists inside scene box
                    max_n_samples_per_batch=-1 if max_n_samples_per_batch == -1 else 2 ** max_n_samples_per_batch,

                    # Sequence
                    n_timesteps=n_timesteps,
                    latent_dim_time=latent_dim_time,  # If hash ensemble is used, this must be n_hash_encodings!

                    # Losses
                    use_masked_rgb_loss=True,
                    alpha_mask_threshold=0,
                    lambda_alpha_loss=lambda_alpha_loss,
                    lambda_near_loss=lambda_near_loss,
                    lambda_empty_loss=lambda_empty_loss,
                    lambda_depth_loss=lambda_depth_loss,
                    lambda_dist_loss=lambda_dist_loss,

                    # Hash Ensemble
                    use_hash_ensemble=use_hash_ensemble,
                    hash_ensemble_config=HashEnsembleConfig(
                        n_hash_encodings=n_hash_encodings,
                        hash_encoding_config=TCNNHashEncodingConfig(),
                        disable_initial_hash_ensemble=True,
                        use_soft_transition=True
                    ),

                    # Deformation Field
                    use_deformation_field=use_deformation_field,
                    use_separate_deformation_time_embedding=True,
                    deformation_field_config=SE3DeformationFieldConfig(
                        warp_code_dim=latent_dim_time_deform,
                        mlp_num_layers=mlp_num_layers,
                        mlp_layer_width=mlp_layer_width,
                    ),
                    disable_occupancy_grid=disable_occupancy_grid,

                    # Scheduler
                    window_hash_encodings_begin=window_hash_encodings_begin,
                    window_hash_encodings_end=window_hash_encodings_end,
                    window_deform_begin=window_deform_begin,
                    window_deform_end=window_deform_end,

                    # View Frustum Culling
                    use_view_frustum_culling=use_view_frustum_culling,
                    view_frustum_culling=view_frustum_culling,
                )
            ),

            optimizers={
                "fields": {
                    "optimizer": AdamOptimizerConfig(lr=lr_main, eps=1e-15, weight_decay=0),
                    "scheduler": StepLRSchedulerConfig(step_size=20000, gamma=8e-1)
                },
                "deformation_field": {
                    "optimizer": AdamOptimizerConfig(lr=lr_deformation_field, eps=1e-15, weight_decay=0),
                    "scheduler": StepLRSchedulerConfig(step_size=20000, gamma=5e-1),
                },
                "embeddings": {
                    "optimizer": AdamOptimizerConfig(lr=lr_embeddings, eps=1e-15, weight_decay=0),
                    "scheduler": StepLRSchedulerConfig(step_size=20000, gamma=8e-1),
                },
            },

            viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
            vis=vis,
        )

        dataparser_config: NeRSembleDataParserConfig = config.pipeline.datamanager.dataparser
        dataparser_config.use_alpha_maps = lambda_alpha_loss > 0
        dataparser_config.use_depth_maps = lambda_empty_loss > 0 or lambda_near_loss > 0 or lambda_depth_loss > 0
        dataparser_config.use_view_frustum_culling = config.pipeline.model.use_view_frustum_culling

        config.set_timestamp()

    # print config
    config.print_to_terminal()

    local_rank = 0
    world_size = 1  # world_size = num_machines * num_gpus_per_machine
    trainer = NeRSembleTrainer(config, local_rank, world_size)
    trainer.setup()

    summary(trainer.pipeline.model)

    # Important to save config after trainer is created, as the trainer fills in some config values
    config.save_config()

    print("DONE setup()")
    trainer.train()
    print("DONE train()")


if __name__ == '__main__':
    tyro.cli(main)
