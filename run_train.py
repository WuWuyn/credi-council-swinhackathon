import subprocess

with open("train_log.txt", "w", encoding="utf-8") as f:
    subprocess.run(["python", "retrain_models.py"], stdout=f, stderr=subprocess.STDOUT)
