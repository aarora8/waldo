from shapely.geometry.polygon import Polygon
import numpy as np


def _evaluate_data(ref_file, hyp_file):

    det_matched = 0
    iou_mat = np.empty([1, 1])
    ref_pols = []
    det_pols = []
    ref_pol_points = []
    det_pol_points = []
    pairs = []
    det_matched_nums = []

    pointlist = _get_pointlist(ref_file)
    for n in range(len(pointlist)):
        points = pointlist[n]
        ref_polygon = _polygon_from_points(points)
        ref_pols.append(ref_polygon)
        ref_pol_points.append(points)

    pointlist = _get_pointlist(hyp_file)
    for n in range(len(pointlist)):
        points = pointlist[n]
        det_polygon = _polygon_from_points(points)
        det_pols.append(det_polygon)
        det_pol_points.append(points)

    if len(ref_pols) > 0 and len(det_pols) > 0:
        output_shape = [len(ref_pols), len(det_pols)]
        iou_mat = np.empty(output_shape)
        ref_rect_mat = np.zeros(len(ref_pols), np.int8)
        det_rect_mat = np.zeros(len(det_pols), np.int8)
        for ref_index in range(len(ref_pols)):
            for hyp_index in range(len(det_pols)):
                polygon_ref = ref_pols[ref_index]
                polygon_hyp = det_pols[hyp_index]
                iou_mat[ref_index, hyp_index] = _get_intersection_over_union(polygon_hyp, polygon_ref)

    for ref_index in range(len(ref_pols)):
        for hyp_index in range(len(det_pols)):
            if ref_rect_mat[ref_index] == 0 and det_rect_mat[hyp_index] == 0:
                if iou_mat[ref_index, hyp_index] > 0.5:
                    ref_rect_mat[ref_index] = 1
                    det_rect_mat[hyp_index] = 1
                    det_matched += 1
                    pairs.append({'reference_data': ref_index, 'det': hyp_index})
                    det_matched_nums.append(hyp_index)

    num_ref = len(ref_pols)
    num_hyp = len(det_pols)
    if num_ref == 0:
        recall = float(1)
        precision = float(0) if num_hyp > 0 else float(1)
    else:
        recall = float(det_matched) / num_ref
        precision = 0 if num_hyp == 0 else float(det_matched) / num_hyp
    h_mean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    per_sample_metrics = dict()
    per_sample_metrics['1'] = {
        'precision': precision,
        'recall': recall,
        'h_mean': h_mean,
        'pairs': pairs,
        'iou_mat': [] if len(det_pols) > 100 else iou_mat.tolist(),
        'gt_pol_points': ref_pol_points,
        'det_pol_points': det_pol_points
    }



def _get_pointlist(ref_file):
    points = []
    return points


def _get_union(p_d, p_g):
    area_a = p_d.area()
    area_b = p_g.area()
    return area_a + area_b - _get_intersection(p_d, p_g)


def _get_intersection_over_union(p_d, p_g):
    try:
        return _get_intersection(p_d, p_g) / _get_union(p_d, p_g)
    except:
        return 0


def _get_intersection(p_d, p_g):
    p_int = p_d & p_g
    if len(p_int) == 0:
        return 0
    return p_int.area()


def _polygon_from_points(points):
    """
    Returns a Polygon object from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return Polygon(point_mat)


def get_score(reference_data, hypothesis_data):

    _validate_data(reference_data, hypothesis_data)
    score = _evaluate_data(reference_data, hypothesis_data)

    return score

