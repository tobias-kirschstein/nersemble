from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from elias.util.random import make_deterministic
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from nersemble.model_manager.nersemble import NeRSembleModelFolder
from nersemble.nerfstudio.config.nersemble_trainer_config import NeRSembleTrainerConfig
from nersemble.nerfstudio.datamanager.nersemble_datamanager import NeRSembleVanillaDataManagerConfig
from nersemble.nerfstudio.dataparser.nersemble_dataparser import NeRSembleDataParserConfig
from nersemble.nerfstudio.engine.nersemble_trainer import NeRSembleTrainer
from nersemble.nerfstudio.engine.step_lr_scheduler import StepLRSchedulerConfig
from nersemble.nerfstudio.field_components.deformation_field import SE3DeformationFieldConfig
from nersemble.nerfstudio.field_components.hash_ensemble import HashEnsembleConfig, TCNNHashEncodingConfig
from nersemble.nerfstudio.models.nersemble_instant_ngp import NeRSembleNGPModelConfig

# from torchinfo import summary

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
    30: [[-2.5, -1.8, -2.5], [2.2, 1.8, 2]],
}


def main(
        participant_id: int,
        sequence_name: str,
        /,
        name: Optional[str] = None,
        vis: Literal['viewer', 'wandb'] = 'wandb',

        lr_main: float = 5e-3,
        lr_deformation_field: float = 1e-3,
        lr_embeddings: float = 5e-3,

        lambda_alpha_loss: float = 1e-2,
        lambda_near_loss: float = 1e-4,
        lambda_empty_loss: float = 1e-2,
        lambda_depth_loss: float = 1e-4,
        lambda_dist_loss: float = 1e-4,
):
    seed = 19980801
    make_deterministic(seed)

    model_folder = NeRSembleModelFolder()
    model_manager = model_folder.new_run(name=name)
    run_name = model_manager.get_run_name()

    # TODO: For some reason Instant NGP only really works if we scale the world coordinate system
    scale_factor = 9
    n_timesteps = 1
    skip_timesteps = 100

    if participant_id in SCENE_BOXES:
        scene_box = torch.tensor(SCENE_BOXES[participant_id]) * scale_factor / 9
    else:
        # Default scene box
        scene_box = torch.tensor([[-2.5, -2, -2.5], [2.5, 3, 2]]) * scale_factor / 9

    config = NeRSembleTrainerConfig(
        project_name="nersemble",
        experiment_name=run_name,
        method_name="nersemble",

        output_dir=Path(model_folder.get_location()),
        run_name=run_name,

        steps_per_eval_batch=500,
        steps_per_eval_image=10000,
        steps_per_eval_all_images=1000000,
        steps_per_save=20000,
        max_num_iterations=100001,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        log_gradients=False,

        pipeline=VanillaPipelineConfig(
            datamanager=NeRSembleVanillaDataManagerConfig(
                dataparser=NeRSembleDataParserConfig(
                    participant_id=participant_id,
                    sequence_name=sequence_name,
                    n_timesteps=n_timesteps,
                    skip_timesteps=skip_timesteps,
                    scale_factor=scale_factor,
                    # [[left, head front, down], [right, head back, up]]
                    scene_box=scene_box,

                    use_view_frustum_culling=False,

                    # auto_scale_poses=False,
                    # center_method='none',
                    # orientation_method='none',

                ),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
                train_num_images_to_sample_from=24,
                train_num_times_to_repeat_images=20,
            ),

            model=NeRSembleNGPModelConfig(
                render_step_size=0.011 * scale_factor / 9.,  # TODO: Is render_step_size correct?
                near_plane=0.05 * scale_factor / 9.,
                far_plane=1e3 * scale_factor / 9.,
                # cone_angle=2e-3, # TODO: Setting to 0 leads to a lot of ray samples in beginning
                alpha_thre=1e-2,  # TODO: Do lower values help? It seems like the occupancy grid is too aggressive
                occ_thre=1e-2,
                early_stop_eps=0,  # Important, otherwise scene may start exploding
                background_color="white",
                grid_levels=1,  # Originally, NeRSemble was trained with a single grid, but larger values also work
                disable_scene_contraction=True,  # To ensure scene only exists inside scene box

                # Sequence
                n_timesteps=n_timesteps,
                latent_dim_time=2,  # If hash ensemble is used, this must be n_hash_encodings!

                # Losses
                use_masked_rgb_loss=True,
                lambda_alpha_loss=lambda_alpha_loss,
                lambda_near_loss=lambda_near_loss,
                lambda_empty_loss=lambda_empty_loss,
                lambda_depth_loss=lambda_depth_loss,
                lambda_dist_loss=lambda_dist_loss,

                # Hash Ensemble
                use_hash_ensemble=False,
                hash_ensemble_config=HashEnsembleConfig(
                    n_hash_encodings=2,
                    hash_encoding_config=TCNNHashEncodingConfig(),
                    disable_initial_hash_ensemble=True,
                    use_soft_transition=True
                ),

                # Deformation Field
                use_deformation_field=False,
                use_separate_deformation_time_embedding=True,
                deformation_field_config=SE3DeformationFieldConfig(
                    warp_code_dim=128,
                    mlp_layer_width=64,
                ),

                # Scheduler
                window_hash_encodings_begin=100,
                window_hash_encodings_end=1000,
                window_deform_begin=1000,
                window_deform_end=10000,
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

    config.set_timestamp()
    # print config
    config.print_to_terminal()

    local_rank = 0
    world_size = 1  # world_size = num_machines * num_gpus_per_machine
    trainer = NeRSembleTrainer(config, local_rank, world_size)
    trainer.setup()

    # summary(trainer.pipeline.model)

    # Important to save config after trainer is created, as the trainer fills in some config values
    config.save_config()

    print("DONE setup()")
    trainer.train()
    print("DONE train()")


if __name__ == '__main__':
    tyro.cli(main)
