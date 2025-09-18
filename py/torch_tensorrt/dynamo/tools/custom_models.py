import torch


def BertModule():
    from transformers import BertModel

    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name, torchscript=True)
    model.eval()
    return model


def BertInputs():
    from transformers import BertTokenizer

    model_name = "bert-base-uncased"
    enc = BertTokenizer.from_pretrained(model_name)
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return [tokens_tensor, segments_tensors]


def StableDiffusion1_4_Unet():
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
    )
    return pipe.unet


def StableDiffusion2_1_Unet():
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    return pipe.unet


def StableDiffusion2_1_VaeDecoder():
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    return pipe.vae.decoder


def MonaiUNet():
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=2,
        in_channels=32,
        out_channels=32,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2,
    )
    return model


def GoogleViTForImageClassification():
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", torch_dtype=torch.float16
    )
    return model
