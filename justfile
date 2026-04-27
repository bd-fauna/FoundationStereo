default:
    @just --list

# Install Python deps via uv (excludes flash-attn)
install:
    uv sync --extra dev

# Install Python deps + flash-attn (requires nvcc / CUDA toolkit)
install-flash:
    uv sync --extra dev --extra flash

# Run the bundled stereo demo
demo:
    uv run python scripts/run_demo.py \
        --left_file ./assets/left.png \
        --right_file ./assets/right.png \
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
