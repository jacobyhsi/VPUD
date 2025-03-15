## Set Up

### Create Environment

Note: can replace `micromamba` with `conda`

```
micromamba create -n vpud python=3.10

micromamba activate vpud

pip install -r requirements.txt
```

### Update Environment
```
pip freeze > requirements.txt
```