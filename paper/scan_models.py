import os, warnings, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model

base_dirs = [
    r"D:/Contest/AI GO/GAN_Test",
    r"D:/Contest/AI GO/Model/Reflection",
]

candidates = {}
for base in base_dirs:
    for folder in sorted(os.listdir(base)):
        fpath = os.path.join(base, folder)
        if not os.path.isdir(fpath):
            continue
        h5s = sorted(glob.glob(os.path.join(fpath, "generator_*.h5")))
        if h5s:
            candidates[folder] = h5s  # all epochs

for name, h5list in sorted(candidates.items()):
    h5path = h5list[-1]  # highest epoch
    try:
        m = load_model(h5path, compile=False)
        types = [type(l).__name__ for l in m.layers]
        lam   = types.count("Lambda")
        mul   = types.count("Multiply")
        params = m.count_params()
        res   = m.input_shape[1]
        ep    = os.path.basename(h5path).replace("generator_","").replace(".h5","")
        all_eps = [os.path.basename(p).replace("generator_","").replace(".h5","") for p in h5list]
        tag   = "SGA     " if lam > 0 else "Baseline"
        print(f"{tag} | {name:<48} | ep={ep:<4} | {res}px | params={params:,} | Lambda={lam} Mul={mul} | avail_eps={all_eps}")
        del m
    except Exception as e:
        print(f"ERROR    | {name}: {e}")
