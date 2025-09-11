import os
import json
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from PIL import Image

N_train = 2000 # Limit to first 2000 images
N_val = 300   # Limit to first 300 images
N_test = 300  # Limit to first 300 images
MSCOCO_PATH = "/share/liyilin-nfs/datasets/MSCOCO"

def generate_train_data(data_path: str, instances: dict, captions: dict):
    i = 0
    for fname in os.listdir(data_path):
        if i == N_train:
            break
        i += 1
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = os.path.splitext(fname)[0][-10:].lstrip("0")
            image = Image.open(os.path.join(data_path, fname)).convert("RGB")
            
            img_instances = instances.get(image_id, [])
            img_captions = captions.get(image_id, [])
            
            yield {
                "images": [image],
                "problem": "<image>Provide a brief description of the given image.",
                "answer": img_captions,
                "instances": img_instances
            }

def generate_val_data(data_path: str, instances: dict, captions: dict):
    fnames = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for fname in fnames[:N_val]:
        image_id = os.path.splitext(fname)[0][-10:].lstrip("0")
        image = Image.open(os.path.join(data_path, fname)).convert("RGB")
        img_instances = instances.get(image_id, [])
        img_captions = captions.get(image_id, [])
        yield {
            "images": [image],
            "problem": "<image>Provide a brief description of the given image.",
            "answer": img_captions,
            "instances": img_instances
        }

def generate_test_data(data_path: str, instances: dict, captions: dict):
    fnames = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for fname in fnames[-N_test:]:
        image_id = os.path.splitext(fname)[0][-10:].lstrip("0")
        image = Image.open(os.path.join(data_path, fname)).convert("RGB")
        img_instances = instances.get(image_id, [])
        img_captions = captions.get(image_id, [])
        yield {
            "images": [image],
            "problem": "<image>Provide a brief description of the given image.",
            "answer": img_captions,
            "instances": img_instances
        }

def load_instances_and_captions(instances_json_path, captions_json_path):
    with open(instances_json_path, 'r') as f:
        instances_data = json.load(f)
    with open(captions_json_path, 'r') as f:
        captions_data = json.load(f)

    instances_map = {}
    for ann in instances_data['annotations']:
        image_id = str(ann['image_id'])
        category_id = ann['category_id']
        if image_id not in instances_map:
            instances_map[image_id] = []
        instances_map[image_id].append(category_id)
    captions_map = {}
    for ann in captions_data['annotations']:
        image_id = str(ann['image_id'])
        caption = ann['caption']
        if image_id not in captions_map:
            captions_map[image_id] = []
        captions_map[image_id].append(caption)
    
    return instances_map, captions_map

def main():
    train_path = f"{MSCOCO_PATH}/train2014"
    val_path = f"{MSCOCO_PATH}/val2014"

    train_instances_json = f"{MSCOCO_PATH}/annotations/instances_train2014.json"
    val_instances_json = f"{MSCOCO_PATH}/annotations/instances_val2014.json"
    train_captions_json = f"{MSCOCO_PATH}/annotations/captions_train2014.json"
    val_captions_json = f"{MSCOCO_PATH}/annotations/captions_val2014.json"

    train_instances_map, train_captions_map = load_instances_and_captions(train_instances_json, train_captions_json)
    val_instances_map, val_captions_map = load_instances_and_captions(val_instances_json, val_captions_json)

    trainset = Dataset.from_generator(
        generate_train_data, 
        gen_kwargs={"data_path": train_path, "instances": train_instances_map, "captions": train_captions_map}
    )
    valset = Dataset.from_generator(
        generate_val_data, 
        gen_kwargs={"data_path": val_path, "instances": val_instances_map, "captions": val_captions_map}
    )
    testset = Dataset.from_generator(
        generate_test_data, 
        gen_kwargs={"data_path": val_path, "instances": val_instances_map, "captions": val_captions_map}
    )

    dataset = DatasetDict({"train": trainset, "validation": valset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("JustinLeeCEO/MSCOCO2014")
    print("Successfully pushed the dataset to HuggingFace!")

if __name__ == "__main__":
    main()

