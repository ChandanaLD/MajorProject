# inspect_graph_v2.py
import torch
from my_models import image_to_graph

dummy = torch.randn(3, 224, 224)
g = image_to_graph(dummy, k=9, patch_size=32, debug=False)

print("GRAPH OBJECT TYPE:", type(g))
if g is None:
    print("image_to_graph returned None")
else:
    # show available attributes
    print("dir(g) sample:", [a for a in dir(g) if not a.startswith("_")][:40])
    # keys() is a method: call it
    try:
        print("g.keys():", g.keys())
    except Exception as e:
        print("g.keys() error:", e)

    # safe attribute prints
    for attr in ("x", "edge_index", "batch", "pos", "num_nodes"):
        try:
            val = getattr(g, attr)
            try:
                print(f"{attr} -> type: {type(val)}, shape/len:", getattr(val, "shape", len(val) if hasattr(val, "__len__") else str(val)))
            except Exception:
                print(f"{attr} -> value: {val}")
        except Exception as e:
            print(f"{attr} -> not present ({e})")

    # more details on x and edge_index if present
    try:
        print("g.x.dtype, g.x.min(), g.x.max():", g.x.dtype, float(g.x.min()), float(g.x.max()))
    except Exception:
        pass
    try:
        print("edge_index sample (first 10 cols):", g.edge_index[:, :10])
    except Exception:
        pass
