conda create -n vpud python=3.10
conda activate vpud
pip install torch accelerate transformers pandas matplotlib datasets scikit-learn flask

python llm.py --llm "llama70b-nemo"
python run.py --data "income" --feature "Education" --shots 3 --sets 10
