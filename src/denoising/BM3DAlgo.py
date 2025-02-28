import bm3d

def applyBM3D(image, noise_std=0.2, stage=bm3d.BM3DStages.HARD_THRESHOLDING) :
    BM3D_denoised_image = bm3d.bm3d(image, sigma_psd=noise_std, stage_arg=stage)
    """
    bm3d library is not well documented yet, but looking into source code....
    sigma_psd - noise standard deviation
    stage_arg: Determines whether to perform hard-thresholding or Wiener filtering.
    stage_arg = BM3DStages.HARD_THRESHOLDING or BM3DStages.ALL_STAGES (slow but powerful)
    All stages performs both hard thresholding and Wiener filtering. 
    """
    return BM3D_denoised_image
