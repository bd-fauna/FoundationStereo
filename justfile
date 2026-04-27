default:
    @just --list

# Install Python deps via uv (excludes flash-attn)
install:
    uv sync --extra dev

# Install Python deps + flash-attn (requires nvcc / CUDA toolkit)
install-flash:
    uv sync --extra dev --extra flash

# Extract paired left/right images + K.txt from a ROS2 bag
extract-bag bag="bag/rosbag2_2026_04_27-17_38_03" out="bag/extracted":
    uv run python scripts/extract_bag.py --bag {{bag}} --out_dir {{out}}

# Run the bundled stereo demo
demo:
    uv run python scripts/run_demo.py \
        --left_file ./assets/left.png \
        --right_file ./assets/right.png \
        --ckpt_dir ./pretrained_models/model_best_bp2.pth \
        --out_dir ./test_outputs/

# Run the demo on the extracted bag pair (run `just extract-bag` first)
demo-bag:
    uv run python scripts/run_demo.py \
        --left_file ./bag/extracted/left.png \
        --right_file ./bag/extracted/right.png \
        --intrinsic_file ./bag/extracted/K.txt \
        --ckpt_dir ./pretrained_models/model_best_bp2.pth \
        --out_dir ./test_outputs/

# Lint with ruff
lint:
    uv run ruff check .

# Auto-fix lint issues
lint-fix:
    uv run ruff check --fix .

# Format with ruff
format:
    uv run ruff format .

# Check formatting without writing
format-check:
    uv run ruff format --check .
