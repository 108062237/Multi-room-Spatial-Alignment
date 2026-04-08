import subprocess
from pathlib import Path


class LGTNetRunner:
    def __init__(self, args):
        self.args = args

        self.repo_root = Path("../LGT-Net")

    def infer(self):
        cfg = self.args.cfg or "src/config/zind.yaml"

        cmd = [
            "python",
            "inference.py",
            "--cfg",
            cfg,
            "--img_glob",
            self.args.img_glob,
            "--output_dir",
            self.args.output_dir,
        ]

        if self.args.post_processing:
            cmd += ["--post_processing", self.args.post_processing]

        print("[LGTNetRunner] Running command:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.repo_root, check=True)
