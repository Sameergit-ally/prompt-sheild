import torch
import transformers


def main():
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
