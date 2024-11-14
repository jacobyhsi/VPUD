```conda create -n vpud python=3.10```

```conda activate vpud```

```pip install torch accelerate transformers pandas matplotlib datasets scikit-learn flask```

Before running llm.py, please fill in your huggingface login token in the file. 

```python llm.py --llm "llama70b-nemo"```

now run run.py:

```python run.py --data "income" --feature "Education" --shots 3 --sets 10```

--data: dataset

--feature: feature you want to vary

--shots: number of icl examples

--sets: number of predictions we want to make with different sets of icl examples.
