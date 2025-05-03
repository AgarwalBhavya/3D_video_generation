from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.default_util import load_config
import torch

def generate_from_text(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('text300M', device)
    config = load_config('diffusion')
    diffusion = diffusion_from_config(config)

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        model_kwargs=dict(prompts=[prompt]),
        guidance_scale=15.0,
        clip_denoised=True,
        use_fp16=torch.cuda.is_available(),
        use_karras=True,
        karras_steps=64,
        sigma_min=0.0001,
        sigma_max=160.0,
        s_churn=0
    )

    for latent in latents:
        mesh = decode_latent_mesh(model, latent)
        with open('output.obj', 'w') as f:
            mesh.write_obj(f)
