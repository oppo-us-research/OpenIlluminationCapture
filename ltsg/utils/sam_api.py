#  created by Isabella Liu (lal005@ucsd.edu)


import os.path as osp
import glob
import imageio
import numpy as np


class SAMAPI:
    predictor = None

    @staticmethod
    def get_instance():
        if SAMAPI.predictor is None:
            sam_checkpoint = "third_party/segment_anything/sam_vit_h_4b8939.pth"
            device = "cuda"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(rgb, mask=None, bbox=None):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        mask: np.ndarray h,w bool
        dbg

        Returns
        -------

        """
        predictor = SAMAPI.get_instance()
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            box=box_input,
            # mask_input=None,
            multimask_output=True,
            return_logits=False,
        )
        maxidx = np.argmax(scores)
        mask = masks[maxidx]
        return mask
