import numpy as np


def calcConstraintViolation(cons, conWgt=None):
    """
    Calculate aggregated constraint violation for each solution.
    """
    if cons is None:
        return None
    cons = np.atleast_2d(np.asarray(cons, dtype=float))
    if conWgt is not None:
        cons = cons * np.atleast_2d(np.asarray(conWgt, dtype=float))
    violation = np.maximum(0.0, cons)
    return np.sum(violation, axis=1)


def isFeasible(cons, conWgt=None):
    """
    Return a boolean mask indicating feasibility of each solution.
    """
    cv = calcConstraintViolation(cons, conWgt)
    if cv is None:
        return None
    return cv <= 0


def compareSolutions(objsA, consA, objsB, consB, conWgt=None):
    """
    Compare two single-objective solutions using Deb's feasibility rules.

    Returns:
        -1 if A is better, 1 if B is better, 0 if tied.
    """
    cvA = calcConstraintViolation(consA, conWgt)
    cvB = calcConstraintViolation(consB, conWgt)

    cvAVal = 0.0 if cvA is None else float(cvA.reshape(-1)[0])
    cvBVal = 0.0 if cvB is None else float(cvB.reshape(-1)[0])
    feasibleA = cvAVal <= 0
    feasibleB = cvBVal <= 0

    if feasibleA and not feasibleB:
        return -1
    if feasibleB and not feasibleA:
        return 1
    if not feasibleA and not feasibleB:
        if cvAVal < cvBVal:
            return -1
        if cvBVal < cvAVal:
            return 1
        return 0

    objA = float(np.asarray(objsA).reshape(-1)[0])
    objB = float(np.asarray(objsB).reshape(-1)[0])
    if objA < objB:
        return -1
    if objB < objA:
        return 1
    return 0


def betterMask(objsA, consA, objsB, consB, conWgt=None):
    """
    Compare two equally-sized sets of single-objective solutions pairwise.

    Returns a boolean mask indicating whether A is better than B.
    """
    objsA = np.atleast_2d(np.asarray(objsA, dtype=float))
    objsB = np.atleast_2d(np.asarray(objsB, dtype=float))
    if objsA.shape != objsB.shape:
        raise ValueError("objsA and objsB must have the same shape.")
    if objsA.shape[1] != 1:
        raise ValueError("betterMask only supports single-objective inputs.")

    cvA = calcConstraintViolation(consA, conWgt)
    cvB = calcConstraintViolation(consB, conWgt)

    if cvA is None:
        cvA = np.zeros(objsA.shape[0], dtype=float)
    if cvB is None:
        cvB = np.zeros(objsB.shape[0], dtype=float)

    feasibleA = cvA <= 0
    feasibleB = cvB <= 0
    better = np.zeros(objsA.shape[0], dtype=bool)

    better[feasibleA & ~feasibleB] = True
    bothInfeasible = ~feasibleA & ~feasibleB
    better[bothInfeasible] = cvA[bothInfeasible] < cvB[bothInfeasible]
    bothFeasible = feasibleA & feasibleB
    better[bothFeasible] = objsA[bothFeasible, 0] < objsB[bothFeasible, 0]
    return better


def argsortSolutions(objs, cons=None, conWgt=None):
    """
    Rank single-objective solutions using Deb's feasibility rules.
    """
    objs = np.atleast_2d(np.asarray(objs, dtype=float))
    if objs.shape[1] != 1:
        raise ValueError("argsortSolutions only supports single-objective inputs.")

    if cons is None:
        return np.argsort(objs.ravel())

    cv = calcConstraintViolation(cons, conWgt)
    feasible = cv <= 0
    keys = np.column_stack((~feasible, cv, objs.ravel()))
    order = np.lexsort(np.flip(keys, axis=1).T)
    return order
