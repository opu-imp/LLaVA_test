import argparse

from modules.predict_utils import eval_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b", help="Model path")
    parser.add_argument("--model_base", type=str, default=None, help="Base model")
    parser.add_argument("--conv_mode", type=str, default=None, help="Conversation mode")
    parser.add_argument("--sep", type=str, default=",", help="Separator")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top p value")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens")
    parser.add_argument("--prompt", type=str, default="What is this image?", help="Prompt")
    parser.add_argument(
        "--image_path", type=str, default="images/demo_cat.jpg", help="Path to the image file"
    )
    args = parser.parse_args()

    outputs = eval_model(
                    args.model_path,
                    args.model_base,
                    args.image_path,
                    args.prompt,
                    args.conv_mode,
                    args.sep,
                    args.temperature,
                    args.top_p,
                    args.num_beams,
                    args.max_new_tokens,
                )
    
    print(outputs)