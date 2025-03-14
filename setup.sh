# Install UV
## Download UV (only once during first-time setup) to appropriate directory where there's sufficient space
## For example: this is the path to my UV directory
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/voyager/projects/jacobyhsi/uv" sh

## Ensure you set UV cache directory in bashrc
vim ~/.bashrc
export UV_CACHE_DIR="/voyager/projects/jacobyhsi/.cache/uv"

# Create UV environment within your working repo
cd vpud
uv venv venv --python 3.10 --seed
source venv/bin/activate

# Install UV packages
uv pip install vllm
uv pip install ipykernel
uv pip install -U ipywidgets
uv pip install nbconvert
uv pip install accelerate
uv pip install openai
uv pip install pandas matplotlib datasets scikit-learn flask
uv pip install gpytorch botorch

# Serve the llm
## Open a screen and serve the LLM. Make sure you Ctrl+C to stop the server before going to bed :D
bash llm.sh

# Run the script (check run.log for run details)
python run.py 2>&1 | tee run.log

# To Zip:
# tar --exclude='*.csv' --exclude='*.pt' --exclude='*.npy' --exclude='.git' --exclude='*.bkp' \
#     --exclude='*.pyc' --exclude='*.zip' --exclude='*.pack' --exclude='*.xls' --exclude='*.data' \
#     --exclude='*.png' --exclude='*.test' --exclude='*.pdf' --exclude='*.pth' \
#     -czvf file_name.tar.gz file_name