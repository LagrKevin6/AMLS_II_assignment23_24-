import subprocess
import os

# Train command
command = [
    'python', 'train_pasd.py',
    '--use_pasd_light',
    '--dataset_name=DIV2K',
    '--pretrained_model_name_or_path=checkpoints/stable-diffusion-v1-5',
    '--output_dir=runs/pasd',
    '--resolution=512',
    '--learning_rate=5e-5',
    '--gradient_accumulation_steps=2',
    '--train_batch_size=2',
    '--tracker_project_name=pasd',
    '--control_type=realisr',
    '--logging_dir=logs',
    '--train_shards_path_or_url=None',
    '--max_train_samples=100000',
    '--num_train_epochs=100',
    '--checkpointing_steps=1000',
    '--mixed_precision=fp16',
    '--dataloader_num_workers=4'
]

def main():
    # Train command
    command_train = [
        'python', 'PASD-main/train_pasd.py',
        '--use_pasd_light',
        '--dataset_name=DIV2K',
        '--pretrained_model_name_or_path=checkpoints/stable-diffusion-v1-5',
        '--output_dir=runs/pasd',
        '--resolution=512',
        '--learning_rate=5e-5',
        '--gradient_accumulation_steps=2',
        '--train_batch_size=2',
        '--tracker_project_name=pasd',
        '--control_type=realisr',
        '--logging_dir=logs',
        '--train_shards_path_or_url=None',
        '--max_train_samples=100000',
        '--num_train_epochs=100',
        '--checkpointing_steps=1000',
        '--mixed_precision=fp16',
        '--dataloader_num_workers=4'
    ]

    subprocess.run(command_train)



    # Image super resolution with PASD**
    # Evaluation
    # 
    pasd_model_path = "runs/pasd/checkpoint-100000" # @param {type: "string"} ["runs/pasd/checkpoint-100000", "runs/pasd_light/checkpoint-120000", "runs/pasd_rrdb/checkpoint-100000", "runs/pasd_light_rrdb/checkpoint-100000"]
    image_path = "DIV2k_val" # @param {type: "string"}
    output_dir = "validation_output" # @param {type: "string"}
    prompt = "clean, high-resolution, 8k" # @param {type: "string"}
    added_prompt = "" # @param {type: "string"}
    negative_prompt = "" #@param {type: "string"}
    upscale = 4 # @param {type: "integer"}
    seed = 36 #@param {type: "integer"}
    added_noise_level = 400 #@param {type: "integer"}
    offset_noise_scale = 0.0 #@param {type: "number"}
    init_latent_with_noise = False #@param {type: "boolean"}
    guidance_scale = 7.5 # @param {type: "number"}
    conditioning_scale = 1.0 # @param {type: "number"}
    blending_alpha = 1.0 # @param {type: "number"}
    multiplier = 0.6 # @param {type: "number"}
    num_inference_steps = 20 # @param {type: "integer"}
    process_size = 768 # @param {type: "integer"}
    decoder_tiled_size = 224 # @param {type: "integer"}
    encoder_tiled_size = 1024 # @param {type: "integer"}
    latent_tiled_size = 320 # @param {type: "integer"}
    latent_tiled_overlap = 8 # @param {type: "integer"}
    mixed_precision = "fp16" # @param {type: "string"} ["no", "fp16", "bf16"]
    control_type = "realisr" # @param {type: "string"} ["realisr", "grayscale"]
    high_level_info = "none" # @param {type: "string"} ["none", "classification", "detection", "caption"]

    os.makedirs(output_dir, exist_ok=True)

    image_path = f'"{image_path}"'
    prompt = f'"{prompt}"'
    added_prompt = f'"{added_prompt}"'
    negative_prompt = f'"{negative_prompt}"'
    upscale = str(upscale)
    guidance_scale = str(guidance_scale)
    conditioning_scale = str(conditioning_scale)
    blending_alpha = str(blending_alpha)
    multiplier = str(multiplier)
    num_inference_steps = str(num_inference_steps)
    process_size = str(process_size)
    decoder_tiled_size = str(decoder_tiled_size)
    encoder_tiled_size = str(encoder_tiled_size)
    latent_tiled_size = str(latent_tiled_size)
    latent_tiled_overlap = str(latent_tiled_overlap)
    high_level_info = "" if high_level_info == "none" else high_level_info
    use_pasd_light = "1" if is_light(pasd_model_path) else "0"
    added_noise_level = str(added_noise_level)
    offset_noise_scale = str(offset_noise_scale)
    init_latent_with_noise = "1" if init_latent_with_noise else "0"
    seed = str(seed)

    command_eval = [
        'python', 'test_pasd.py',
        '--pasd_model_path', pasd_model_path,
        '--image_path', image_path,
        '--output_dir', output_dir,
        '--control_type', control_type,
        '--high_level_info', high_level_info,
        '--prompt', prompt,
        '--added_prompt', added_prompt,
        '--negative_prompt', negative_prompt,
        '--upscale', upscale,
        '--guidance_scale', guidance_scale,
        '--mixed_precision', mixed_precision,
        '--blending_alpha', blending_alpha,
        '--multiplier', multiplier,
        '--num_inference_steps', num_inference_steps,
        '--process_size', process_size,
        '--decoder_tiled_size', decoder_tiled_size,
        '--encoder_tiled_size', encoder_tiled_size,
        '--latent_tiled_size', latent_tiled_size,
        '--latent_tiled_overlap', latent_tiled_overlap,
        '--added_noise_level', added_noise_level,
        '--offset_noise_scale', offset_noise_scale,
        '--use_pasd_light', use_pasd_light,
        '--init_latent_with_noise', init_latent_with_noise,
        '--seed', seed
    ]

    subprocess.run(command_eval)

if __name__ == "__main__":
    main()

