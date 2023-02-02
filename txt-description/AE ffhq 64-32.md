# FFHQ AE 64->32x32x3
config file: models/first_stage_models/vq-f2/vq-f2-64.yaml

model:
  base_learning_rate: 4.5e-06
  target: ldm.models.autoencoder.VQModel
  params:
    embed_dim: 3
    n_embed: 8192
    monitor: val/rec_loss
    ddconfig:
      double_z: false
      z_channels: 3
      resolution: 64
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult:
        - 1
        - 2
      num_res_blocks: 2
      attn_resolutions: []
      dropout: 0.0
    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: false
        disc_in_channels: 3
        disc_start: 0
        disc_weight: 0.75
        codebook_weight: 1.0

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 16
    num_workers: 2
    wrap: true
    train:
      target: taming.data.faceshq.FFHQTrain
      params:
        size: 64
    validation:
      target: taming.data.faceshq.FFHQValidation
      params:
        size: 64

Train checkpoint & train log & img generation directory
zhren@128.32.116.228:/home/zhren/Charlie/charlie-latent-diffusion/latent-diffusion/logs/2023-01-31T21-11-20_vq-f2-64/