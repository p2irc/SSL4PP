"""Average Domain Accuracy (ADA) metric for wheat head detection task.""" ""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import _fix_empty_tensors, _input_validator
from torchvision.ops import box_convert


class AverageDomainAccuracy(Metric):
    """Compute the average domain accuracy for wheat head detection task.

    Args:
        box_format: tr, optional
            The format of the boxes. Defaults to "xyxy".
        compute_on_step: Optional[bool]
            ``forward `` only calls ``update()`` and return None if this is set
            to False.
        dist_sync_on_step: Optional[bool]
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.

    Attributes:
        detection_boxes:
            List of tensors containing the detection boxes.
        groundtruth_boxes:
            List of tensors containing the ground truth boxes.
        groundtruth_domains:
            List of tensors containing the ground truth domains.

    """

    detection_boxes: List[Tensor]
    groundtruth_boxes: List[Tensor]
    groundtruth_domains: List[Tensor]

    def __init__(
        self,
        box_format: str = "xyxy",
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Init method."""
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format

        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_domains", default=[], dist_reduce_fx=None)

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:
        """Update the metric states.

        Args:
            preds: (List[Dict[str, Tensor]])
                List of dictionaries containing the predictions.
            target: (List[Dict[str, Tensor]])
                List of dictionaries containing the ground truth.

        """
        _input_validator(preds, target)

        for item in preds:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
            self.detection_boxes.append(boxes)

        for item in target:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
            self.groundtruth_boxes.append(boxes)
            self.groundtruth_domains.append(item["domain"])

    @staticmethod
    def _accuracy(dts: Tensor, gts: Tensor, iou_thr: int = 0.5) -> float:
        """Compute accuracy between two tensors.

        Accuracy is defined as the ratio of the number of true positives to the
        number of true positives plus the number of false positives plus the
        number of false negative. The expected format is (x_min, y_min, x_max,
        y_max)

        Args:
            dts (Tensor): Detection boxes.
            gts (Tensor): Ground truth boxes.
            iou_thr (float, optional): IoU threshold. Defaults to 0.5.

        Returns:
            acc: float
                Accuracy score.

        """
        if len(dts) > 0 and len(gts) > 0:
            pick = AverageDomainAccuracy._get_matches(dts, gts, overlapThresh=iou_thr)
            tp = len(pick)
            fn = len(gts) - len(pick)
            fp = len(dts) - len(pick)
            acc = float(tp) / (float(tp) + float(fn) + float(fp))
        elif len(dts) == 0 and len(gts) > 0:
            acc = 0.0
        elif len(dts) > 0 and len(gts) == 0:
            acc = 0.0
        elif len(dts) == 0 and len(gts) == 0:
            acc = 1.0

        return acc

    @staticmethod
    def _get_matches(
        gts: ArrayLike, dts: ArrayLike, overlapThresh: Optional[float] = 0.5
    ) -> List:
        """Compute matches between groundtruth and detections.

        Args:
            gts: ArrayLike
                Groundtruth boxes.
            dts: ArrayLike
                Detection boxes.
            overlapThresh: float, optional
                Overlap threshold. Defaults to 0.5.

        Returns:
            pick: List
                List of matches.

        """
        gts = np.array([np.array(bbox) for bbox in gts])
        boxes = np.array([np.array(bbox) for bbox in dts])

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # keep looping while some indexes still remain in the indexes
        # list
        area_gt = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        gts = gts[np.argsort(area_gt)]
        idxs = list(range(len(area)))
        for (x, y, xx, yy) in gts:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            area_ = (xx - x) * (yy - y)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x, x1[idxs])
            yy1 = np.maximum(y, y1[idxs])
            xx2 = np.minimum(xx, x2[idxs])
            yy2 = np.minimum(yy, y2[idxs])

            # compute the width and height of the bounding box
            ww = np.maximum(0, xx2 - xx1 + 1)
            hh = np.maximum(0, yy2 - yy1 + 1)

            # compute intersection over union (union is area 1 +area 2-intersection)
            overlap = (ww * hh) / (area[idxs] + area_ - (ww * hh))

            # true_matches = np.where(overlap > overlapThresh)
            if len(overlap) > 0:
                potential_match = np.argmax(overlap)  # we select the best match

                if (
                    overlap[potential_match] > overlapThresh
                ):  # we check if it scores above the threshold
                    pick.append(idxs[potential_match])
                    # delete all indexes from the index list that have
                    idxs = np.delete(idxs, [potential_match])

        # return only the bounding boxes that were picked using the
        # integer data type
        return pick

    def compute(self):
        """Compute the average domain accuracy."""
        self._move_list_states_to_cpu()

        df = pd.DataFrame(columns=["domain", "acc"])
        for idx in range(len(self.groundtruth_boxes)):
            acc = self._accuracy(self.detection_boxes[idx], self.groundtruth_boxes[idx])
            df.loc[idx] = [self.groundtruth_domains[idx].item(), acc]

        domain_acc_score = df.groupby("domain").mean()
        ada_score = domain_acc_score.mean().values[0]

        # NOTE: for degbugging
        print(domain_acc_score)

        return round(ada_score, 4)
