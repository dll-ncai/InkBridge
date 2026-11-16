
import json
nb=json.load(open("$U/exp/FMD/xs/weights/github/cleandift/notebooks/get_correspondences.ipynb"))
for c in nb["cells"]:
    if c["cell_type"]=="code": print("".join(c["source"])); print("-----")
