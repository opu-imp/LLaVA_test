import argparse
import os
import json
import random
from pathlib import Path
from tqdm import tqdm

from modules.text_utils import text_splitter, check_yes_no
from modules.predict_utils import eval_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument(
        "--dataset_dir", type=str, default="../../scrapping_new_species/"
    )
    parser.add_argument("--json_path", type=str, default="animal_dataset/labels")
    parser.add_argument(
        "--saved_dir", type=str, default="./outputs/animal_dataset/test5/"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    json_dir = Path(args.json_path)
    saved_dir = Path(args.saved_dir)
    json_files = [f for f in os.listdir(dataset_dir / json_dir) if f.endswith(".json")]
    random.seed(42)
    random.shuffle(json_files)
    chunks = [
        json_files[i : i + args.chunk_size]
        for i in range(0, len(json_files), args.chunk_size)
    ]

    for i, chunk in tqdm(enumerate(chunks)):
        question_template = "Please answer “Yes” or “No” if the description is correct for the image.\n\n"
        correct_image_path = []
        sentences_list = []
        predicted_list = []

        for j, filename in enumerate(chunk):
            with open(dataset_dir / json_dir / filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                sentences = text_splitter(data.get("prompt", ""))
                sentences_list.append(sentences)

        for j, filename in enumerate(chunk):
            if j == 0:
                continue

            correct_rate_list = []
            with open(dataset_dir / json_dir / filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_file = Path(data.get("image_path", ""))

            for sentences in sentences_list:
                correct_score = 0
                for sentence in sentences:
                    outputs = eval_model(
                        args.model_path,
                        args.model_base,
                        str(dataset_dir / image_file),
                        question_template + sentence,
                        args.conv_mode,
                        args.sep,
                        args.temperature,
                        args.top_p,
                        args.num_beams,
                        args.max_new_tokens,
                    )
                    if check_yes_no(outputs) == 1:
                        print(sentence)
                    correct_score += check_yes_no(outputs)
                correct_rate = correct_score / len(sentences)
                correct_rate_list.append(correct_rate)

            correct_image_path.append(
                str(dataset_dir / image_file).replace(".png", ".jpg")
            )
            predicted_list.append(correct_rate_list.index(max(correct_rate_list)))

        with open(saved_dir / f"results-{i}.json", "w") as f:
            json.dump(
                {
                    "correct_number": correct_image_path,
                    "predicted_list": predicted_list,
                },
                f,
                indent=2,
            )

        break
