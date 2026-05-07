import copy

import numpy as np

from .core import NDSort, crowdingDist, calcConstraintViolation, argsortSolutions


class Population:
    def __init__(self, decs, objs=None, cons=None, conWgt=None):
        self.conWgt = conWgt
        self.decs = np.atleast_2d(np.copy(decs))
        self.objs = None if objs is None else np.atleast_2d(np.copy(objs))
        self.cons = None if cons is None else np.atleast_2d(np.copy(cons))

        self.nPop, self.D = self.decs.shape
        self.nOutput = None if self.objs is None else self.objs.shape[1]
        self.frontNo = None
        self.crowdDis = None

    @property
    def isEvaluated(self):
        return self.objs is not None

    @property
    def hasCons(self):
        return self.cons is not None

    def requireEvaluated(self):
        if not self.isEvaluated:
            raise ValueError("Population is not evaluated yet.")

    def requireConsistentEvalState(self, otherPop):
        if self.isEvaluated != otherPop.isEvaluated:
            raise ValueError("Cannot combine evaluated and unevaluated populations.")
        if self.hasCons != otherPop.hasCons:
            raise ValueError("Cannot combine populations with inconsistent constraint state.")
        if self.conWgt is None and otherPop.conWgt is None:
            return
        if self.conWgt is None or otherPop.conWgt is None:
            raise ValueError("Cannot combine populations with inconsistent constraint weights.")
        if not np.array_equal(self.conWgt, otherPop.conWgt):
            raise ValueError("Cannot combine populations with different constraint weights.")

    def copy(self):
        return copy.deepcopy(self)

    def add(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Population):
            otherPop = args[0]
        elif len(args) >= 1:
            decs = args[0]
            objs = args[1] if len(args) >= 2 else kwargs.get("objs", None)
            cons = args[2] if len(args) >= 3 else kwargs.get("cons", None)
            otherPop = Population(decs, objs, cons, self.conWgt)
        else:
            raise TypeError("add() expects a Population or (decs, objs=None, cons=None)")

        self.requireConsistentEvalState(otherPop)
        self.decs = np.vstack((self.decs, otherPop.decs))
        if self.isEvaluated:
            self.objs = np.vstack((self.objs, otherPop.objs))
            if self.hasCons:
                self.cons = np.vstack((self.cons, otherPop.cons))

        self.nPop = self.decs.shape[0]
        self.frontNo = None
        self.crowdDis = None
        return self

    def merge(self, otherPop):
        return self.add(otherPop)

    def merged(self, otherPop):
        newPop = self.copy()
        newPop.merge(otherPop)
        return newPop

    def getBest(self, k: int | None = None):
        self.requireEvaluated()
        if self.nOutput == 1:
            return self._bestSingle(k)
        return self._bestMulti(k)

    def _bestSingle(self, k: int | None = None):
        args = self.argsort()
        idx = args[:k] if k is not None else args[:1]
        return Population(
            self.decs[idx],
            self.objs[idx],
            self.cons[idx] if self.cons is not None else None,
            self.conWgt,
        )

    def _bestMulti(self, k: int | None = None):
        frontNo = self.frontNo
        if self.cons is not None:
            CV = calcConstraintViolation(self.cons, self.conWgt)
            feasible = CV <= 0
            feasiblePop = self[feasible]
            if len(feasiblePop) > 0:
                feasibleFrontNo = feasiblePop.frontNo
                if feasibleFrontNo is None:
                    feasibleFrontNo, _ = NDSort(feasiblePop.objs, feasiblePop.cons)
                crowDis = feasiblePop.crowdDis
                if crowDis is None:
                    crowDis = crowdingDist(feasiblePop.objs, feasibleFrontNo)
                feasiblePop.frontNo = feasibleFrontNo
                feasiblePop.crowdDis = crowDis
                bestPop = feasiblePop[feasibleFrontNo == 1]
            else:
                sortedIdx = np.argsort(CV)
                kk = 10 if k is None else k
                return self[sortedIdx[:kk]]
        else:
            if frontNo is None:
                frontNo, _ = NDSort(self.objs, self.cons)
            bestPop = self[frontNo == 1]

        if k is not None and len(bestPop) > k:
            bestFrontNo = bestPop.frontNo
            if bestFrontNo is None:
                bestFrontNo, _ = NDSort(bestPop.objs, bestPop.cons)
            bestCrowdDis = bestPop.crowdDis
            if bestCrowdDis is None:
                bestCrowdDis = crowdingDist(bestPop.objs, bestFrontNo)
            bestPop.frontNo = bestFrontNo
            bestPop.crowdDis = bestCrowdDis
            sortedIdx = np.lexsort((-bestCrowdDis, bestFrontNo))
            bestPop = bestPop[sortedIdx[:k]]

        return bestPop

    def getParetoFront(self):
        self.requireEvaluated()
        return self.getBest(k=None)

    def argsort(self):
        self.requireEvaluated()
        if self.nOutput == 1:
            return argsortSolutions(self.objs, self.cons, self.conWgt)

        frontNo, _ = NDSort(self.objs, self.cons)
        crowDis = crowdingDist(self.objs, frontNo)
        return np.lexsort((-crowDis, frontNo))

    def clip(self, lb, ub):
        self.decs = np.clip(self.decs, lb, ub, out=self.decs)
        return self

    def replace(self, index, pop):
        self.requireConsistentEvalState(pop)
        self.decs[index, :] = pop.decs
        if self.isEvaluated:
            self.objs[index, :] = pop.objs
            if self.hasCons:
                self.cons[index, :] = pop.cons
        self.frontNo = None
        self.crowdDis = None
        return self

    def assignEval(self, objs, cons=None):
        self.objs = np.atleast_2d(np.copy(objs))
        self.cons = None if cons is None else np.atleast_2d(np.copy(cons))
        self.nOutput = self.objs.shape[1]
        self.frontNo = None
        self.crowdDis = None
        return self

    def size(self):
        return self.nPop, self.D

    def __getitem__(self, index):
        if isinstance(index, (slice, list, np.ndarray)):
            decs = self.decs[index]
            objs = self.objs[index] if self.objs is not None else None
            cons = self.cons[index] if self.cons is not None else None
        elif isinstance(index, (int, np.integer)):
            decs = self.decs[index:index + 1]
            objs = self.objs[index:index + 1] if self.objs is not None else None
            cons = self.cons[index:index + 1] if self.cons is not None else None
        else:
            raise TypeError("Index must be int, slice, list, or ndarray")
        return Population(decs, objs, cons, self.conWgt)

    def __len__(self):
        return self.nPop
