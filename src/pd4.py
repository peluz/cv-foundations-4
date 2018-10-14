import argparse


parser = argparse.ArgumentParser(
    description="Image Segmentation")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--r1", action="store_true",
                   help="Requisito 1")
group.add_argument("--r2", action="store_true",
                   help="Requisito 2")


def main(r1, r2):
    if r1:
        pass
    elif r2:
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2)
