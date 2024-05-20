from model import LightWeightGan

def get_model(params):
    GAN = LightWeightGan(
            latent_dim = params["latent_dim"],
            attn_res_layers = params["attn_res_layers"],
            freq_chan_attn = params["freq_chan_attn"],
            image_size = params["image_size"],
            fmap_max = params["fmap_max"],
            disc_output_size = params["disc_output_size"],
            transparent = params["transparent"],
            greyscale = params["greyscale"],
        )
    return GAN


def initialize_model_and_params(model, params):
    GAN = get_model(params)
    model = GAN.G
    return model
