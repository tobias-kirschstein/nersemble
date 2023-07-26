from collections import defaultdict
from typing import Optional, Tuple, Dict

import numpy as np
import tinycudann as tcnn
import torch
from jaxtyping import Shaped
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import (
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead, FieldHeadNames,
)
from nerfstudio.field_components.spatial_distortions import (
    SpatialDistortion,
)
from nerfstudio.fields.base_field import shift_directions_for_tcnn
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from torch import Tensor

from nersemble.nerfstudio.field_components.hash_ensemble import HashEnsembleConfig, HashEnsemble
from nersemble.util.chunker import chunked


class NeRSembleNeRFactoField(TCNNNerfactoField):

    def __init__(
            self,
            aabb: Tensor,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_levels: int = 16,
            max_res: int = 2048,
            log2_hashmap_size: int = 19,
            num_layers_color: int = 3,
            num_layers_transient: int = 2,
            hidden_dim_color: int = 64,
            hidden_dim_transient: int = 64,
            appearance_embedding_dim: int = 32,
            transient_embedding_dim: int = 16,
            use_transient_embedding: bool = False,
            use_semantics: bool = False,
            num_semantic_classes: int = 100,
            pass_semantic_gradients: bool = False,
            use_pred_normals: bool = False,
            use_average_appearance_embedding: bool = False,
            spatial_distortion: Optional[SpatialDistortion] = None,

            # NeRSemble additions
            use_appearance_embedding: bool = False,
            spherical_harmonics_degree: int = 4,
            use_hash_ensemble: bool = False,
            hash_ensemble_config: Optional[HashEnsembleConfig] = None,
            max_n_samples_per_batch: int = -1
    ) -> None:
        # "Jump" over super class __init__() and directly call super super class __init__()
        super(TCNNNerfactoField, self).__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding

        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients

        # NeRSemble additions
        self.use_hash_ensemble = use_hash_ensemble
        self.max_n_samples_per_batch = max_n_samples_per_batch

        base_res: int = 16
        features_per_level: int = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        # ----------------------------------------------------------
        # Direction encoding
        # ----------------------------------------------------------

        if spherical_harmonics_degree > 0:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": spherical_harmonics_degree,
                },
            )
        else:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Identity"
                },
            )

        # ----------------------------------------------------------
        # Base network with hash encoding
        # ----------------------------------------------------------

        if use_hash_ensemble:
            self.hash_ensemble = HashEnsemble(hash_ensemble_config)
            # Hash encoding is computed seperately, so base MLP just takes inputs without adding encoding
            base_network_encoding_config = {
                "otype": "Identity",
                "n_dims_to_encode": self.hash_ensemble.get_out_dim(),
            }
            base_network_n_input_dims = self.hash_ensemble.get_out_dim()
        else:
            base_network_encoding_config = {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            }
            base_network_n_input_dims = 3

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=base_network_n_input_dims,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config=base_network_encoding_config,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # ----------------------------------------------------------
        # RGB head
        # ----------------------------------------------------------

        n_input_dims = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if self.use_appearance_embedding:
            n_input_dims += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        # ----------------------------------------------------------
        # Other heads not used by NeRSemble
        # ----------------------------------------------------------
        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient - 1,
                },
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.n_output_dims)

        # semantics
        if self.use_semantics:
            self.mlp_semantics = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.position_encoding.n_output_dims,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

    def density_fn(self,
                   positions: Shaped[Tensor, "*bs 3"],
                   times: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   window_hash_encodings: Optional[float] = None,
                   time_codes: Optional[torch.Tensor] = None) -> Shaped[Tensor, "*bs 1"]:

        del times
        # Need to figure out a better way to describe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            metadata={"time_codes": time_codes}
        )

        density, _ = self.get_density(ray_samples, window_hash_encodings=window_hash_encodings)
        return density

    def get_density(self, ray_samples: RaySamples, window_hash_encodings: float) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        max_chunk_size = len(positions) if self.max_n_samples_per_batch == -1 else self.max_n_samples_per_batch

        time_codes = ray_samples.metadata['time_codes'] if 'time_codes' in ray_samples.metadata else None

        densities = []
        base_mlp_outs = []
        for positions_chunked, time_codes_chunked in chunked(max_chunk_size, positions, time_codes):

            # Make sure the tcnn gets inputs between 0 and 1.
            selector = ((positions_chunked > 0.0) & (positions_chunked < 1.0)).all(dim=-1)
            positions_chunked = positions_chunked * selector[..., None]
            self._sample_locations = positions_chunked
            if not self._sample_locations.requires_grad:
                self._sample_locations.requires_grad = True
            positions_flat = positions_chunked.view(-1, 3)

            # Hash Ensemble encoding
            if self.use_hash_ensemble:
                # Run 3d points through hash ensemble to get blended meaningful spatial features for base MLP to decode
                base_inputs = self.hash_ensemble(positions_flat,
                                                 conditioning_code=time_codes_chunked,
                                                 window_hash_encodings=window_hash_encodings,
                                                 )
            else:
                base_inputs = positions_flat

            h = self.mlp_base(base_inputs).view(*positions_chunked.shape[:-1], -1)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            self._density_before_activation = density_before_activation

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            density = trunc_exp(density_before_activation.to(positions_chunked))
            density = density * selector[..., None]

            densities.append(density)
            base_mlp_outs.append(base_mlp_out)

        densities = torch.cat(densities, dim=0)
        base_mlp_outs = torch.cat(base_mlp_outs, dim=0)

        return densities, base_mlp_outs

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        # Only change to nerfstudio is adding a toggle for appearance dim

        assert density_embedding is not None

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        positions = ray_samples.frustums.get_positions()

        max_chunk_size = len(ray_samples) if self.max_n_samples_per_batch == -1 else self.max_n_samples_per_batch

        outputs = defaultdict(list)
        for directions_chunked, camera_indices_chunked, density_embedding_chunked, positions_chunked in chunked(max_chunk_size, directions_flat, camera_indices, density_embedding, positions):
            outputs_shape = directions_chunked.shape[:-1]
            d = self.direction_encoding(directions_chunked)

            # appearance
            embedded_appearance = None
            if self.use_appearance_embedding:
                if self.training:
                    embedded_appearance = self.embedding_appearance(camera_indices_chunked)
                else:
                    if self.use_average_appearance_embedding:
                        embedded_appearance = torch.ones(
                            (*camera_indices_chunked.shape[:-1], self.appearance_embedding_dim), device=directions.device
                        ) * self.embedding_appearance.mean(dim=0)
                    else:
                        embedded_appearance = torch.zeros(
                            (*camera_indices_chunked.shape[:-1], self.appearance_embedding_dim), device=directions.device
                        )

            # transients
            if self.use_transient_embedding and self.training:
                embedded_transient = self.embedding_transient(camera_indices_chunked)
                transient_input = torch.cat(
                    [
                        density_embedding_chunked.view(-1, self.geo_feat_dim),
                        embedded_transient.view(-1, self.transient_embedding_dim),
                    ],
                    dim=-1,
                )
                x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
                outputs[FieldHeadNames.UNCERTAINTY].append(self.field_head_transient_uncertainty(x))
                outputs[FieldHeadNames.TRANSIENT_RGB].append(self.field_head_transient_rgb(x))
                outputs[FieldHeadNames.TRANSIENT_DENSITY].append(self.field_head_transient_density(x))

            # semantics
            if self.use_semantics:
                semantics_input = density_embedding_chunked.view(-1, self.geo_feat_dim)
                if not self.pass_semantic_gradients:
                    semantics_input = semantics_input.detach()

                x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
                outputs[FieldHeadNames.SEMANTICS].append(self.field_head_semantics(x))

            # predicted normals
            if self.use_pred_normals:
                positions_flat = self.position_encoding(positions_chunked.view(-1, 3))
                pred_normals_inp = torch.cat([positions_flat, density_embedding_chunked.view(-1, self.geo_feat_dim)], dim=-1)

                x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
                outputs[FieldHeadNames.PRED_NORMALS].append(self.field_head_pred_normals(x))

            rgb_inputs = [d, density_embedding_chunked.view(-1, self.geo_feat_dim)]
            if self.use_appearance_embedding:
                rgb_inputs.append(embedded_appearance)

            h = torch.cat(rgb_inputs, dim=-1)

            rgb_chunked = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.RGB].append(rgb_chunked)

        for key, chunked_output in outputs.items():
            outputs[key] = torch.cat(chunked_output, dim=0)

        return outputs

    def forward(self, ray_samples: RaySamples,
                compute_normals: bool = False,
                window_hash_encodings: Optional[float] = None) -> Dict[FieldHeadNames, Tensor]:

        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples, window_hash_encodings=window_hash_encodings)
        else:
            density, density_embedding = self.get_density(ray_samples, window_hash_encodings=window_hash_encodings)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
