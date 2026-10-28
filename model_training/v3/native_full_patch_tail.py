"""Full-parameter counterpart to the adapted-patch Q/V adapter candidate."""
from native_adapted_patch_tail import NativeAdaptedPatchTail,NativePatchClinicalHead
from native_full_tail import NativeFullTail


class NativeFullPatchTail(NativeAdaptedPatchTail):
    def __init__(self,backbone,head):
        NativeFullTail.__init__(self,backbone,head)
