import argparse
from utils.text_to_3d import generate_from_text

def main():
    parser = argparse.ArgumentParser(description="Generate 3D object from text prompt")
    parser.add_argument("--text", type=str, required=True, help="Text prompt for 3D generation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save .obj files")
    args = parser.parse_args()

    print(f"[INFO] Prompt: {args.text}")
    generate_from_text(args.text, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
