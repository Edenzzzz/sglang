import json, sys
import numpy as np
for f in sys.argv[1:]:
    v = [json.loads(l).get("itl_slo") for l in open(f) if '"itl_slo"' in l]
    v = [x for x in v if x]
    print(f.split("/")[-2][:50], len(v), np.percentile(v, [10, 50, 90]).round(1))
