import subprocess
from pathlib import Path


class HorizonNetRunner:
    def __init__(self, args):
        self.args = args
        self.repo_root = Path("../HorizonNet")

    def infer(self):
        ckpt = self.args.ckpt or "ckpt/resnet50_rnn__zind.pth"

        cmd = [
            "python",
            "inference.py",
            "--pth",
            ckpt,
            "--img_glob",
            self.args.img_glob,
            "--output_dir",
            self.args.output_dir,
        ]

        if self.args.visualize:
            cmd.append("--visualize")

        print("[HorizonNetRunner] Running command:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.repo_root, check=True)
