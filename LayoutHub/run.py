import argparse
from models.registry import MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified runner for indoor layout models"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Which model to use (e.g., horizonnet, lgtnet)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["infer"],
        help="For now only support inference",
    )

    # 通用參數
    parser.add_argument("--cfg", help="Config file (for lgtnet)")
    parser.add_argument("--ckpt", help="Checkpoint path (for horizonnet)")
    parser.add_argument("--img_glob", required=True, help="Input image glob")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # model-specific 選項
    parser.add_argument(
        "--post_processing",
        help="Post-processing mode (for lgtnet, e.g., manhattan)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize (for horizonnet)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    RunnerClass = MODEL_REGISTRY[args.model]
    runner = RunnerClass(args)

    if args.mode == "infer":
        runner.infer()
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented yet.")


if __name__ == "__main__":
    main()
