# Image-Segmentation

```python
from torch import nn
from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair

class EdgeConv(nn.Module):
    def __init__(self, in_feat, out_feat, batch_norm=False, allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata["x"] = h_src
            g.dstdata["x"] = h_dst
            g.apply_edges(fn.v_sub_u("x", "x", "theta"))
            g.edata["theta"] = self.theta(g.edata["theta"])
            g.dstdata["phi"] = self.phi(g.dstdata["x"])
            if not self.batch_norm:
                g.update_all(fn.e_add_v("theta", "phi", "e"), fn.max("e", "x"))
            else:
                g.apply_edges(fn.e_add_v("theta", "phi", "e"))
                g.edata["e"] = self.bn(g.edata["e"])
                g.update_all(fn.copy_e("e", "e"), fn.max("e", "x"))
            return g.dstdata["x"]
```
