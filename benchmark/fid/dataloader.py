import os
from datasets import Features, Value, Sequence, Dataset
from PIL import Image
import io

features = Features({
    'image': {
        'bytes': Value('binary'),
        'path': Value('string'),
    },
    'filepath': Value(dtype='string', id=None),
    'sentids': Sequence(Value(dtype='int32', id=None)),
    'filename': Value(dtype='string', id=None),
    'imgid': Value(dtype='int32', id=None),
    'split': Value(dtype='string', id=None),
    'sentences_tokens': Sequence(Sequence(Value(dtype='string', id=None))),
    'sentences_raw': Sequence(Value(dtype='string', id=None)),
    'sentences_sentid': Sequence(Value(dtype='int32', id=None)),
    'cocoid': Value(dtype='int32', id=None),
})

def get_dataset():
    parquet_dir = "/workspace/xDiT/.cache/huggingface/datasets/HuggingFaceM4___coco/validation_files"
    validation_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.startswith("coco-validation") and f.endswith(".parquet")]
    dataset = Dataset.from_parquet(validation_files, features=features)
    return dataset

def process_image(example):
    try:
        image_data = example["image"]['bytes']
        try:
            example["image"] = Image.open(io.BytesIO(image_data))
        except Exception as e:
            print(f"Error processing image: {e}")
            example["image"] = None
        return example
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
