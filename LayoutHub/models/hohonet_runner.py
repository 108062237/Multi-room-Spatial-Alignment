import subprocess
from pathlib import Path


class HoHoNetRunner:
    def __init__(self, args):
        self.args = args
        self.repo_root = Path("../HoHoNet")

    def infer(self):
        # Default config and checkpoint if not provided
        # Updated to correct config path found in repo
        cfg = self.args.cfg or "config/Zind_layout/HOHO_layout_zind_resnet34.yaml"
        ckpt = self.args.ckpt or "ckpt/Zind_layout_HOHO_layout_zind_resnet34/best.pth" 
        
        # Check if ckpt exists, if not maybe it's named differently?
        # Listing ckpt dir might be good idea. But let's try updating config first.
        
        cmd = [
            "python",
            "infer_layout.py",
            "--cfg",
            cfg,
            "--pth",
            ckpt,
            "--inp",
            self.args.img_glob,
            "--out",
            self.args.output_dir,
        ]

        print("[HoHoNetRunner] Running command:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.repo_root, check=True)
