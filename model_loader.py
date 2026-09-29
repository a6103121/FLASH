import os
import torch
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    SYSTEM_MESSAGE
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model

def prepare_llava_inputs(template, query, image, tokenizer):
    # template = SYSTEM_MESSAGE + ' ' + template
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)
    assistant_trigger = " ASSISTANT:"

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_system = (
        tokenizer(
            SYSTEM_MESSAGE,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )

    token_assistant = (
        tokenizer(
            assistant_trigger,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )


    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    sys_start_idx = 1
    sys_end_idx = sys_start_idx + len(token_system[0])

    instruction_start_idx = sys_end_idx

    img_start_idx = instruction_start_idx + len(token_before[0])
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH

    instruction_end_idx = img_end_idx + len(token_after[0])

    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_system, token_before, image_token, token_after,token_assistant], dim=1)

    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids

    # no image query with <ImageHere>, img_start_idx, img_end_idx, {"image": image_tensor, "input_ids": no image query token with -200}
    return qu, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs

class ModelLoader:
    def __init__(self, model_name,args):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model(args)
    def load_model(self,args):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser(args.model_path)
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            # no image query with <ImageHere>, img_start_idx, img_end_idx, {"image": image_tensor, "input_ids": no image query token with -200}
            questions, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_llava_inputs(
                template, query, image, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx
        self.sys_start_idx = sys_start_idx
        self.sys_end_idx = sys_end_idx
        self.instruction_start_idx = instruction_start_idx
        self.instruction_end_idx = instruction_end_idx

        self.input = kwargs


        return questions, kwargs

    def decode(self, output_ids):
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text
