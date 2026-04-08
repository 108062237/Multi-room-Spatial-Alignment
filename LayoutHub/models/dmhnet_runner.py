import subprocess
from pathlib import Path


class DMHNetRunner:
    def __init__(self, args):
        self.args = args
        self.repo_root = Path("../DMH-Net")

    def infer(self):
        # Default config and checkpoint if not provided
        cfg = self.args.cfg or "cfgs/matterport.yaml"
        ckpt = self.args.ckpt or "ckpt/matterport_v1.pth"

        cmd = [
            "/home/wang/anaconda3/envs/dmhnet/bin/python",
            "inference.py",
            "--cfg_file",
            cfg,
            "--ckpt",
            ckpt,
            "--input_file",
            self.args.img_glob,
            "--visu_path",
            self.args.output_dir,
            "--save_json",
        ]

        print("[DMHNetRunner] Running command:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.repo_root, check=True)
