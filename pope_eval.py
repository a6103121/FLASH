import argparse
import json
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attention import llama_modify
from constants import INSTRUCTION_TEMPLATE, POPE_CHAT_PATH, SYSTEM_MESSAGE
from eval_data_loader import POPEChatDataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm

def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    disable_torch_init()
    
    
parser = argparse.ArgumentParser(description="POPE evaluation on LLaVA-1.5.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--pope-type", type=str, help="model")
parser.add_argument("--data-path",type=str,default=None,help="data path")
parser.add_argument("--model-path",type=str,default=None,help="model path")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--max-tokens", type=int, default=1)

parser.add_argument("--lambda_s", type=float, default=0.9)
parser.add_argument("--lambda_v", type=float, default=1.2)
parser.add_argument("--k", type=float, default=5)
parser.add_argument("--tau", type=float, default=0.2)
parser.add_argument("--s-start-layer", type=int, default=15)
parser.add_argument("--s-end-layer", type=int, default=26)
parser.add_argument("--v-start-layer", type=int, default=2)
parser.add_argument("--v-end-layer", type=int, default=32)

args = parser.parse_known_args()[0]

setup_seeds()

model_loader = ModelLoader(args.model,args=args)

args.pope_path = POPE_CHAT_PATH[args.pope_type]
pope_dataset = POPEChatDataSet(
    pope_path=args.pope_path,
    data_path=args.data_path,
    trans=model_loader.image_processor,
)

pope_loader = torch.utils.data.DataLoader(
    pope_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=32,
    drop_last=False,
)

base_dir = "./pope/" + args.model
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


file_parts = [
    f"pope_eval_{args.pope_type}",
    f"_lambdas_{args.lambda_s}",
    f"_lambdav_{args.lambda_v}" ,
    f"_tau_{args.tau}" ,
    f"_k_{args.k}"
]


file_name = "".join(file_parts)
template = INSTRUCTION_TEMPLATE[args.model]
if args.model == "llava-1.5":
    template = SYSTEM_MESSAGE + template


for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):

    image = data["image"]
    queries = np.array(data["query"])
    label = torch.stack(data["label"])
    kwargs = {}

    round = label.size()[0]

    for idx in range(round):
        query = queries[idx, :]
        query = query.tolist()
        lal = label[idx, :].tolist()

        # prepare inputs for model
        questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

        llama_modify(
            model = model_loader.llm_model,
            s_start_layer = args.s_start_layer,
            s_end_layer = args.s_end_layer,
            v_start_layer = args.v_start_layer,
            v_end_layer = args.v_end_layer,
            lambda_v = args.lambda_v,
            lambda_s = args.lambda_s,
            tau = args.tau,
            k = args.k,
            img_start_idx = model_loader.img_start_idx,
            img_end_idx = model_loader.img_end_idx
        )

        with torch.inference_mode():
            outputs = model_loader.llm_model.generate(
                do_sample=args.sample,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                num_beams=args.beam,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                **kwargs,
            )

        output_text = model_loader.decode(outputs)

        for i in range(len(output_text)):
            with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
                json.dump(
                    {
                        "query": query[i],
                        "label": lal[i],
                        "ans": output_text[i],
                        "question": questions[i],
                        "file_path": file_name,
                    },
                    f,
                )
                f.write("\n")
