from ._distance import meanNearestDistance


def IGD(popObjs, optimum):
    return meanNearestDistance(optimum, popObjs)
