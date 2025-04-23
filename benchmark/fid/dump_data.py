import json
import os
from argparse import ArgumentParser
from PIL import Image
import io
from tqdm import tqdm, trange
from dataloader import get_dataset, process_image

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_root", type=str, default="./coco")
    args = parser.parse_args()

    dataset = get_dataset()

    prompt_list = []
    for i in trange(len(dataset["sentences_raw"])):
        prompt = dataset["sentences_raw"][i][i % len(dataset["sentences_raw"][i])]
        prompt_list.append(prompt)

    os.makedirs(args.output_root, exist_ok=True)
    prompt_path = os.path.join(args.output_root, "prompts.json")
    with open(prompt_path, "w") as f:
        json.dump(prompt_list, f, indent=4)

    os.makedirs(os.path.join(args.output_root, "images"), exist_ok=True)

    dataset = get_dataset()
    dataset_map = dataset.map(process_image, num_proc=4, batched=False)

    for i, image in enumerate(tqdm(dataset_map["image"])):
        if isinstance(image, dict):
            temp = image["bytes"]
            image = Image.open(io.BytesIO(temp))
        image.save(os.path.join(args.output_root, "images", f"{i:05}.png"))
    
    # for i, image in enumerate(tqdm(dataset["image"])):
    #     image.save(os.path.join(args.output_root, "images", f"{i:04}.png"))
