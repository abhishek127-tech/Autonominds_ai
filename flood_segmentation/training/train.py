import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Flood segmentation training placeholder")
    parser.parse_args()
    parser.error(
        "training is not implemented; define a licensed dataset format, model, "
        "and evaluation protocol before training"
    )


if __name__ == "__main__":
    main()