from collections import namedtuple
from shapely.geometry.polygon import Polygon
import numpy as np
from PIL import Image

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def get_score(reference_data, hypothesis_data):

    validate_data(reference_data, hypothesis_data)
    score = evaluate_data(reference_data, hypothesis_data)

    return score


def evaluate_data(reference_data, hypothesis_data):
    perSampleMetrics = {}
    matchedSum = 0
    gt = reference_data
    subm = hypothesis_data

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    for result_file in gt:
        gt_file = gt
        subm_file = subm
        detMatched = 0
        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        pairs = []
        detMatchedNums = []

        pointlist = get_pointlist(gt_file)
        for n in range(len(pointlist)):
            points = pointlist[n]
            gtRect = Rectangle(*points)
            gtPol = rectangle_to_polygon(gtRect)
            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

        if result_file in subm:

            pointlist = get_pointlist(subm_file)
            for n in range(len(pointlist)):
                points = pointlist[n]
                detRect = Rectangle(*points)
                detPol = rectangle_to_polygon(detRect)
                detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        if iouMat[gtNum, detNum] > 0.5:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        numGtCare = len(gtPols)
        numDetCare = len(detPols)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

    # Compute MAP and MAR

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
    methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics}

    return resDict


def get_pointlist(gt_file):
    points = []
    return points


def validate_data(reference_data, hypothesis_data):
    points = []
    validate_clockwise_points(points)
    return


def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. "
            "Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the "
            "standard one, with the image origin at the upper left, the X axis extending to the right and Y axis "
            "extending downwards.")


def polygon_from_points(points):
    """
    Returns a Polygon object from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return Polygon(pointMat)


def rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)

    pointMat = resBoxes[0].reshape([2, 4]).T

    return Polygon(pointMat)


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

