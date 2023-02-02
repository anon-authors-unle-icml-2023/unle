import abc

from flax.struct import PyTreeNodeMeta, __dataclass_transform__


@__dataclass_transform__()
class ABCPyTreeNodeMeta(abc.ABCMeta, PyTreeNodeMeta):
    pass
