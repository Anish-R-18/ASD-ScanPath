"""Times one sampler call at several CPU thread counts and checks the output does not depend on the thread count."""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze"]
import torch
from omegaconf import OmegaConf
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.generate_cond import sample_conditioned
from training.external_gaze4asd_loader import StimulusBank
from training.eval_external_v2 import IMG_DIR, BASE_CFG
dev = torch.device("cpu")
st = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
b2, _ = mu.build_model(BASE_CFG, "training/runs/baseline/best_model.pth", dev, st["mean_log_inter_time"], st["std_log_inter_time"], used_cfg_out=os.environ["TEMP"] + "/timing.yaml")
p2 = ConditionedTPPGaze(b2, 512).to(dev); p2.load_state_dict(torch.load("training/runs/phase2_embedding_only/best_model_conditioned.pth", map_location=dev)); mu._fix_unregistered_device(p2.base, dev); p2.eval()
img = StimulusBank(IMG_DIR).tensor("top_image_1.png")
feat = {}
orig = p2.base.get_metadata
def cached(imgs):
    if "f" not in feat:
        feat["f"] = orig(imgs[:1])
    return feat["f"].expand(imgs.shape[0], *feat["f"].shape[1:])
p2.base.get_metadata = cached
ref = None
for th in (1, 2, 4, 8, 16):
    torch.set_num_threads(th)
    torch.manual_seed(101001)
    t = time.time(); out = sample_conditioned(p2, img, 0, t_end=3.0, n_sim=20, temperature=1.0); dt = time.time() - t
    same = ref is None or (out[0].shape == ref[0].shape and float((out[1] - ref[1]).abs().max()) < 1e-5 and torch.equal(out[2], ref[2]))
    ref = ref or out
    print(f"threads {th:2d}: {dt:6.2f} s per call; steps {out[0].shape[1]}; output identical to 1-thread run: {same}", flush=True)
