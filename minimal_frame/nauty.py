"""
Graph Automorphism Algorithm
============

Description:
------------
Python implementation of nauty algorithm based on nauty-2.8.8 at https://pallini.di.uniroma1.it/

The implementation mainly covers files among
    - nauty.h
    - nauty.c
    - nautil.c
    - naugraph.c

Please note that nauty is covered by its own licensing terms. All rights of above files are reserved to Brendan McKay under Apache License 2.0.

Author:
-------
Yuchao Lin

"""
import numpy as np
from numba import njit
from sympy.combinatorics import Permutation, PermutationGroup

MAX_STACK = 100
NAUTY_MAX_LAYERS = 2000000002


@njit(cache=True)
def HASH(l: int, i: int) -> int:
    return ((l ^ 0o65435) + i) & 0o77777


@njit(cache=True)
def CLEANUP(l: int) -> int:
    return l % 0o77777


@njit(cache=True)
def bestcell(n: int, g: np.ndarray, lab: np.ndarray, ptn: np.ndarray, level: int) -> int:
    # Find non-singleton cells and put their starts in workperm
    workperm = np.zeros(n, dtype=np.int32)
    bucket = np.zeros(n, dtype=np.int32)

    nnt = 0
    i = 0

    while i < n:
        if ptn[i] > level:
            workperm[nnt] = i
            nnt += 1
            while ptn[i] > level:
                i += 1
        i += 1

    if nnt == 0:
        return n

    # Initialize bucket to zero
    bucket[:nnt] = 0

    gp_values = g[lab[workperm[:nnt]], :]

    # Count the number of non-trivial neighbors for each non-singleton cell
    for v2 in range(1, nnt):
        workset = set()
        i = workperm[v2] - 1
        while True:
            i += 1
            workset.add(lab[i])
            if ptn[i] <= level:
                break

        for v1 in range(0, v2):
            gp = gp_values[v1]

            setword1 = False
            setword2 = False

            for j in workset:
                if gp[j]:
                    setword1 = True
                else:
                    setword2 = True
                if setword1 and setword2:
                    break

            if setword1 and setword2:
                bucket[v1] += 1
                bucket[v2] += 1

    v1 = np.argmax(bucket[:nnt])
    return workperm[v1]


def doref(n: int, g: np.ndarray, lab: np.ndarray, ptn: np.ndarray, level: int, numcells: int,
          invar: np.ndarray, active: np.ndarray, minlev: int,
          maxlev: int, invararg: int, invarproc=None):
    workset = np.zeros(n, dtype=np.bool_)

    if np.any(active):
        tvpos = np.nonzero(active)[0][0]
    else:
        tvpos = 0

    numcells, code = refine(n, g, lab, ptn, level, numcells, invar, active)

    if invarproc and numcells < n and minlev <= level <= maxlev:
        invarproc(tvpos, invar, invararg)
        active[:] = False

        for i in range(n):
            workset[i] = invar[lab[i]]

        nc = numcells
        cell1 = 0

        while cell1 < n:
            pw = workset[cell1]
            same = True

            cell2 = cell1
            while ptn[cell2] > level:
                if workset[cell2] != pw:
                    same = False
                cell2 += 1

            if not same:
                sorted_indices = np.argsort(workset[cell1:cell2 + 1])
                workset[cell1:cell2 + 1], lab[cell1:cell2 + 1] = workset[cell1:cell2 + 1][sorted_indices], \
                    lab[cell1:cell2 + 1][sorted_indices]

                for i in range(cell1 + 1, cell2 + 1):
                    if workset[i] != workset[i - 1]:
                        ptn[i - 1] = level
                        numcells += 1
                        active[i] = True

            cell1 = cell2 + 1

        if numcells > nc:
            qinvar = 2
            longcode = code
            numcells, code = refine(n, g, lab, ptn, level, numcells, invar, active)
            longcode = HASH(longcode, code)
            code = CLEANUP(longcode)
        else:
            qinvar = 1
    else:
        qinvar = 0
    return qinvar, numcells, code


@njit(cache=True)
def refine(n: int,
           g: np.ndarray,
           lab: np.ndarray,
           ptn: np.ndarray,
           level: int,
           numcells: int,
           count: np.ndarray,
           active: np.ndarray
           ):
    workperm = np.zeros(n, dtype=np.int32)
    bucket = np.zeros(n + 2, dtype=np.int32)
    longcode = numcells
    hint = 0

    while numcells < n:
        split1 = -1
        if hint != 0 and active[hint]:
            split1 = hint
        else:
            for pos in range(hint, n):
                if active[pos]:
                    split1 = pos
                    break

            if split1 == -1:
                for pos in range(hint):
                    if active[pos]:
                        split1 = pos
                        break

        if split1 == -1:
            break

        active[split1] = False

        split2 = split1
        while ptn[split2] > level:
            split2 += 1

        longcode = HASH(longcode, split1 + split2)
        if split1 == split2:  # trivial splitting
            gptr = g[lab[split1], :]
            cell1 = 0
            while cell1 < n:
                cell2 = cell1
                while ptn[cell2] > level:
                    cell2 += 1

                if cell1 != cell2:
                    c1, c2 = cell1, cell2
                    while c1 <= c2:
                        labc1 = lab[c1]
                        if gptr[labc1]:
                            c1 += 1
                        else:
                            lab[c1], lab[c2] = lab[c2], labc1
                            c2 -= 1

                    if c2 >= cell1 and c1 <= cell2:
                        ptn[c2] = level
                        longcode = HASH(longcode, c2)
                        numcells += 1

                        if active[cell1] or c2 - cell1 >= cell2 - c1:
                            active[c1] = True
                            if c1 == cell2:
                                hint = c1
                        else:
                            active[cell1] = True
                            if c2 == cell1:
                                hint = cell1

                cell1 = cell2 + 1
        else:  # nontrivial splitting
            cnts = np.sum(g[:, lab[split1:split2 + 1]], axis=1)
            longcode = HASH(longcode, split2 - split1 + 1)

            cell1 = 0
            while cell1 < n:
                cell2 = cell1
                while ptn[cell2] > level:
                    cell2 += 1

                if cell1 == cell2:
                    cell1 = cell2 + 1
                    continue

                cnt = cnts[lab[cell1]]
                count[cell1] = cnt
                bmin, bmax = cnt, cnt
                bucket[cnt] = 1

                for i in range(cell1 + 1, cell2 + 1):
                    cnt = cnts[lab[i]]
                    count[i] = cnt

                    while bmin > cnt:
                        bmin -= 1
                        bucket[bmin] = 0

                    while bmax < cnt:
                        bmax += 1
                        bucket[bmax] = 0

                    bucket[cnt] += 1

                if bmin == bmax:
                    longcode = HASH(longcode, bmin + cell1)
                    cell1 = cell2 + 1
                    continue

                c1 = cell1
                maxcell = -1
                for i in range(bmin, bmax + 1):
                    if bucket[i] != 0:
                        c2 = c1 + bucket[i]
                        bucket[i] = c1
                        longcode = HASH(longcode, i + c1)
                        if c2 - c1 > maxcell:
                            maxcell = c2 - c1
                            maxpos = c1

                        if c1 != cell1:
                            active[c1] = True
                            if c2 - c1 == 1:
                                hint = c1
                            numcells += 1

                        if c2 <= cell2:
                            ptn[c2 - 1] = level

                        c1 = c2

                for i in range(cell1, cell2 + 1):
                    workperm[bucket[count[i]]] = lab[i]
                    bucket[count[i]] += 1
                for i in range(cell1, cell2 + 1):
                    lab[i] = workperm[i]

                if not active[cell1]:
                    active[cell1] = True
                    active[maxpos] = False

                cell1 = cell2 + 1

    longcode = HASH(longcode, numcells)
    longcode = CLEANUP(longcode)
    return numcells, longcode


@njit(cache=True)
def isautom(n: int, g: np.ndarray, perm: np.ndarray, digraph: bool = False) -> bool:
    for i in range(n):
        pg = g[i, :]
        pgp = g[perm[i], :]

        for pos in np.where(pg)[0]:
            if pos <= i and not digraph:
                continue
            posp = perm[pos]
            if not pgp[posp]:
                return False
    return True


@njit(cache=True)
def cheapautom(n: int, ptn: np.ndarray, level: int, digraph: bool = False) -> bool:
    if digraph:
        return False

    k = n
    nnt = 0
    i = 0
    while i < n:
        k -= 1
        if ptn[i] > level:
            nnt += 1
            while ptn[i] > level:
                i += 1
        i += 1

    return k <= nnt + 1 or k <= 4


@njit(cache=True)
def permset(set1, perm):
    set2 = np.zeros_like(set1, dtype=np.bool_)
    set2[perm[set1]] = True
    return set2


def putstring(s, value):
    return s + value


def writeperm(perm, n):
    s = ""
    curlen = 0
    workperm = [0] * n

    for i in range(n - 1, -1, -1):
        workperm[i] = 0

    for i in range(n):
        if workperm[i] == 0 and perm[i] != i:
            l = i
            intlen = len(str(l))
            s = putstring(s, '(')
            curlen += 1
            while True:
                s = putstring(s, str(l))
                curlen += intlen + 1
                k = l
                l = perm[l]
                workperm[k] = 1
                if l != i:
                    intlen = len(str(l))
                    s = putstring(s, ' ')
                else:
                    break
            s = putstring(s, ')')
            curlen += 1

    if curlen == 0:
        s = putstring(s, "(1)\n")
    else:
        s = putstring(s, '\n')

    return s


class Nauty:
    def __init__(self, n, g, lab=None, ptn=None, active=None, defaultptn=True, getcanon=True, digraph=False,
                 write_perm=False,
                 tc_level=100, mininvarlevel=0, maxinvarlevel=1):
        assert n < 2 ** 32, "currently no support for too many nodes"

        self.n = n
        self.worktop = 2 * n * MAX_STACK
        self.mininvarlevel = mininvarlevel
        self.maxinvarlevel = maxinvarlevel

        self.g = g
        self.canong = np.zeros_like(g)
        self.perms = []
        # GLOBAL ARRAYS
        self.fixedpts = np.zeros(n, dtype=np.bool_)

        self.firstlab = np.zeros(n, dtype=np.int32)
        self.canonlab = np.zeros(n, dtype=np.int32)
        self.firstcode = np.zeros(n + 2, dtype=np.int32)
        self.canoncode = np.zeros(n + 2, dtype=np.int32)
        self.firsttc = np.zeros(n + 2, dtype=np.int32)

        self.orbits = np.array(range(n), dtype=np.int32)

        # GLOBAL VARIABLES
        self.gca_first = 0
        self.gca_canon = 0
        self.noncheaplevel = 1
        self.allsamelevel = 0
        self.eqlev_first = 0
        self.eqlev_canon = -1
        self.comp_canon = 0
        self.samerows = 0
        self.canonlevel = 0
        self.stabvertex = 0
        self.cosetindex = 0

        # INVAR
        self.invar = np.zeros(n, dtype=np.int32)
        self.invararg = 0

        self.workspace = np.zeros(self.worktop, dtype=np.bool_)
        self.fmptr = 0

        self.stats = {
            'maxlevel': 0,
            'grpsize1': 1.0,
            'grpsize2': 0.0,
            'numgenerators': 0,
            'numnodes': 0,
            'numbadleaves': 0,
            'tctotal': 0,
            'canupdates': 0,
            'numorbits': n,
        }

        self.getcanon = getcanon
        self.digraph = digraph
        self.write_perm = write_perm
        self.needshortprune = False

        if digraph:
            self.tc_level = 0
        else:
            self.tc_level = tc_level

        if defaultptn:
            lab = np.array(range(n), dtype=np.int32)
            ptn = np.full(n, fill_value=NAUTY_MAX_LAYERS, dtype=np.int32)
            ptn[n - 1:] = 0
            numcells = 1

            self.active = np.zeros(n, dtype=np.bool_)
            self.active[0] = True
        else:
            ptn[n - 1] = 0
            ptn[ptn != 0] = NAUTY_MAX_LAYERS
            numcells = np.sum(ptn == 0)
            if active is None:
                self.active = np.zeros(n, dtype=np.bool_)
                i = 0
                while i < n:
                    self.active[i] = True
                    while ptn[i]:
                        i += 1
                    i += 1
            else:
                self.active = active

        self.firstpathnode(lab, ptn, 1, numcells)
        if self.getcanon:
            self.updatecan(self.g, self.canong, self.canonlab, self.samerows)
            lab[:] = self.canonlab[:]

    def size(self):
        return int(self.stats['grpsize1'] * 10 ** self.stats['grpsize2'])

    def generate_full_group(self):
        G = PermutationGroup([Permutation(perm.tolist()) for perm in self.perms])
        return list(G.generate())

    def processnode(self, lab, ptn, level, numcells):
        code = 0
        workperm = np.zeros(self.n, dtype=np.int32)

        if self.eqlev_first != level and (not self.getcanon or self.comp_canon < 0):
            code = 4
        elif numcells == self.n:
            if self.eqlev_first == level:
                for i in range(self.n):
                    workperm[self.firstlab[i]] = lab[i]
                if self.gca_first >= self.noncheaplevel or isautom(self.n, self.g, workperm, self.digraph):
                    code = 1
            if code == 0:
                if self.getcanon:
                    sr = 0
                    if self.comp_canon == 0:
                        if level < self.canonlevel:
                            self.comp_canon = 1
                        else:
                            self.updatecan(self.g, self.canong, self.canonlab, self.samerows)
                            self.samerows = self.n
                            self.comp_canon, sr = self.testcanlab(self.g, self.canong, lab)

                    if self.comp_canon == 0:
                        for i in range(self.n):
                            workperm[self.firstlab[i]] = lab[i]
                        code = 2
                    elif self.comp_canon > 0:
                        code = 3
                    else:
                        code = 4

        if code != 0 and level > self.stats['maxlevel']:
            self.stats['maxlevel'] = level

        if code == 0:
            return level
        elif code == 1:
            if self.fmptr == self.worktop:
                self.fmptr -= 2 * self.n
            self.fmperm(workperm, self.fmptr, self.fmptr + self.n)
            self.fmptr += 2 * self.n
            self.perms.append(workperm)
            if self.write_perm:
                print(writeperm(workperm, self.n))
            self.stats['numorbits'] = self.orbjoin(self.orbits, workperm)
            self.stats['numgenerators'] += 1
            return self.gca_first

        elif code == 2:
            if self.fmptr == self.worktop:
                self.fmptr -= 2 * self.n
            self.fmperm(workperm, self.fmptr, self.fmptr + self.n)
            self.fmptr += 2 * self.n
            save = self.stats['numorbits']
            self.stats['numorbits'] = self.orbjoin(self.orbits, workperm)
            if save == self.stats['numorbits']:
                if self.gca_canon != self.gca_first:
                    self.needshortprune = True
                return self.gca_canon
            self.perms.append(workperm)
            if self.write_perm:
                print(writeperm(workperm, self.n))
            self.stats['numgenerators'] += 1

            if self.orbits[self.cosetindex] < self.cosetindex:
                return self.gca_first

            if self.gca_canon != self.gca_first:
                self.needshortprune = True
            return self.gca_canon

        elif code == 3:
            self.stats['canupdates'] += 1
            self.canonlab[:] = lab[:]
            self.canonlevel = self.eqlev_canon = self.gca_canon = level
            self.comp_canon = 0
            self.canoncode[level + 1] = 0o077777
            self.samerows = sr

        elif code == 4:
            self.stats['numbadleaves'] += 1

        if level != self.noncheaplevel:
            ispruneok = True
            if self.fmptr == self.worktop:
                self.fmptr -= 2 * self.n
            self.fmptn(lab, ptn, self.noncheaplevel, self.fmptr, self.fmptr + self.n)
            self.fmptr += 2 * self.n
        else:
            ispruneok = False

        save = self.allsamelevel - 1 if self.allsamelevel > self.eqlev_canon else self.eqlev_canon
        newlevel = self.noncheaplevel - 1 if self.noncheaplevel <= save else save

        if ispruneok and newlevel != self.gca_first:
            self.needshortprune = True

        return newlevel

    def firstpathnode(self, lab, ptn, level, numcells):
        tcell = np.zeros(self.n, dtype=np.bool_)
        self.stats['numnodes'] += 1
        qinvar, numcells, code = doref(self.n, self.g, lab, ptn, level, numcells, self.invar, self.active,
                                       self.mininvarlevel, self.maxinvarlevel, self.invararg)

        self.firstcode[level] = code
        tc = -1
        if numcells != self.n:
            tcellsize, tc = self.maketargetcell(self.g, lab, ptn, level, tcell, self.tc_level, -1)
            self.stats['tctotal'] += tcellsize
        self.firsttc[level] = tc

        if numcells == self.n:
            self.firstterminal(lab, level)
            return level - 1

        if self.noncheaplevel >= level and not cheapautom(self.n, ptn, level, self.digraph):
            self.noncheaplevel = level + 1

        nonzeros = np.nonzero(tcell)[0]
        if len(nonzeros) > 0:
            tv1 = tv = nonzeros[0]
        else:
            tv1 = tv = -1

        index = 0
        while tv >= 0:
            if self.orbits[tv] == tv:
                self.breakout(lab, ptn, level + 1, tc, tv, self.active)
                self.fixedpts[tv] = True
                self.cosetindex = tv

                if tv == tv1:
                    rtnlevel = self.firstpathnode(lab, ptn, level + 1, numcells + 1)
                    childcount = 1
                    self.gca_first = level
                    self.stabvertex = tv1
                else:
                    rtnlevel = self.othernode(lab, ptn, level + 1, numcells + 1)
                    childcount += 1

                self.fixedpts[tv] = False

                if rtnlevel < level:
                    return rtnlevel

                if self.needshortprune:
                    self.needshortprune = False
                    tcell = self.shortprune(tcell, self.workspace[self.fmptr - self.n:self.fmptr])

                self.recover(ptn, level)

            if self.orbits[tv] == tv1:
                index += 1

            ind = -1
            for i in range(tv + 1, len(tcell)):
                if tcell[i]:
                    ind = i
                    break
            tv = ind

        self.stats['grpsize1'] *= index
        if self.stats['grpsize1'] >= 1e10:
            self.stats['grpsize1'] /= 1e10
            self.stats['grpsize2'] += 10

        if tcellsize == index and self.allsamelevel == level + 1:
            self.allsamelevel -= 1

        return level - 1

    def othernode(self, lab, ptn, level, numcells):
        tcell = np.zeros(self.n, dtype=np.bool_)
        self.stats['numnodes'] += 1

        qinvar, numcells, code = doref(self.n, self.g, lab, ptn, level, numcells, self.invar, self.active,
                                       self.mininvarlevel, self.maxinvarlevel, self.invararg)

        if self.eqlev_first == level - 1 and code == self.firstcode[level]:
            self.eqlev_first = level

        if self.getcanon:
            if self.eqlev_canon == level - 1:
                if code < self.canoncode[level]:
                    self.comp_canon = -1
                elif code > self.canoncode[level]:
                    self.comp_canon = 1
                else:
                    self.comp_canon = 0
                    self.eqlev_canon = level

            if self.comp_canon > 0:
                self.canoncode[level] = code

        tc = -1

        if numcells < self.n and (self.eqlev_first == level or (self.getcanon and self.comp_canon >= 0)):
            if not self.getcanon or self.comp_canon < 0:
                tcellsize, tc = self.maketargetcell(self.g, lab, ptn, level, tcell, self.tc_level, self.firsttc[level])
                if tc != self.firsttc[level]:
                    self.eqlev_first = level - 1
            else:
                tcellsize, tc = self.maketargetcell(self.g, lab, ptn, level, tcell, self.tc_level, -1)
            self.stats['tctotal'] += tcellsize

        rtnlevel = self.processnode(lab, ptn, level, numcells)

        if rtnlevel < level:
            return rtnlevel

        if self.needshortprune:
            self.needshortprune = False
            tcell = self.shortprune(tcell, self.workspace[self.fmptr - self.n:self.fmptr])

        if not cheapautom(self.n, ptn, level, self.digraph):
            self.noncheaplevel = level + 1

        nonzeros = np.nonzero(tcell)[0]
        if len(nonzeros) > 0:
            tv1 = tv = nonzeros[0]
        else:
            tv1 = tv = -1

        while tv >= 0:
            self.breakout(lab, ptn, level + 1, tc, tv, self.active)
            self.fixedpts[tv] = True
            rtnlevel = self.othernode(lab, ptn, level + 1, numcells + 1)

            self.fixedpts[tv] = False

            if rtnlevel < level:
                return rtnlevel

            if self.needshortprune:
                self.needshortprune = False
                tcell = self.shortprune(tcell, self.workspace[self.fmptr - self.n:self.fmptr])

            if tv == tv1:
                tcell = self.longprune(tcell, self.fixedpts)

            self.recover(ptn, level)

            ind = -1
            for i in range(tv + 1, len(tcell)):
                if tcell[i]:
                    ind = i
                    break
            tv = ind

        return level - 1

    def firstterminal(self, lab: np.ndarray, level: int):
        self.stats['maxlevel'] = level
        self.gca_first = self.allsamelevel = self.eqlev_first = level
        self.firstcode[level + 1] = 0o77777
        self.firsttc[level + 1] = -1

        for i in range(self.n):
            self.firstlab[i] = lab[i]

        if self.getcanon:
            self.canonlevel = self.eqlev_canon = self.gca_canon = level
            self.comp_canon = 0
            self.samerows = 0
            for i in range(self.n):
                self.canonlab[i] = lab[i]
            for i in range(level + 1):
                self.canoncode[i] = self.firstcode[i]
            self.canoncode[level + 1] = 0o77777
            self.stats['canupdates'] = 1

    def updatecan(self, g: np.ndarray, canong: np.ndarray, lab: np.ndarray, samerows: int):
        workperm = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            workperm[lab[i]] = i

        for i in range(samerows, self.n):
            canong[i, :] = permset(g[lab[i], :], workperm)

    def testcanlab(self, g: np.ndarray, canong: np.ndarray, lab: np.ndarray) -> (int, int):
        workperm = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            workperm[lab[i]] = i

        for i in range(self.n):
            workset = permset(g[lab[i], :], workperm)
            ph = canong[i, :]

            for j in range(self.n):
                if workset[j] < ph[j]:
                    samerows = i
                    return -1, samerows
                elif workset[j] > ph[j]:
                    samerows = i
                    return 1, samerows

        samerows = self.n
        return 0, samerows

    def recover(self, ptn: np.ndarray, level: int):
        ptn[ptn > level] = NAUTY_MAX_LAYERS

        if level < self.noncheaplevel:
            self.noncheaplevel = level + 1
        if level < self.eqlev_first:
            self.eqlev_first = level

        if self.getcanon:
            if level < self.gca_canon:
                self.gca_canon = level
            if level <= self.eqlev_canon:
                self.eqlev_canon = level
                self.comp_canon = 0

    def orbjoin(self, orbits: np.ndarray, map_arr: np.ndarray) -> int:
        for i in range(self.n):
            if map_arr[i] != i:
                j1 = orbits[i]
                while orbits[j1] != j1:
                    j1 = orbits[j1]
                j2 = orbits[map_arr[i]]
                while orbits[j2] != j2:
                    j2 = orbits[j2]

                if j1 < j2:
                    orbits[j2] = j1
                elif j1 > j2:
                    orbits[j1] = j2

        j1 = 0
        for i in range(self.n):
            orbits[i] = orbits[orbits[i]]
            if orbits[i] == i:
                j1 += 1

        return j1

    def fmperm(self, perm: np.ndarray, fix: int, mcr: int):
        self.workspace[fix:fix + self.n] = False
        self.workspace[mcr:mcr + self.n] = False

        workset = np.zeros(self.n, dtype=np.bool_)

        for i in range(self.n):
            if perm[i] == i:
                self.workspace[fix + i] = True
                self.workspace[mcr + i] = True
            elif not workset[i]:
                l = i
                while True:
                    k = l
                    l = perm[l]
                    workset[k] = True
                    if l == i:
                        break
                self.workspace[mcr + i] = True

    def fmptn(self, lab: np.ndarray, ptn: np.ndarray, level: int, fix: int, mcr: int):
        self.workspace[fix:fix + self.n] = False
        self.workspace[mcr:mcr + self.n] = False

        i = 0
        while i < self.n:
            if ptn[i] <= level:
                self.workspace[fix + lab[i]] = True
                self.workspace[mcr + lab[i]] = True
            else:
                lmin = lab[i]
                i += 1
                if lab[i] < lmin:
                    lmin = lab[i]
                while ptn[i] > level:
                    i += 1
                    if lab[i] < lmin:
                        lmin = lab[i]
                self.workspace[mcr + lmin] = True
            i += 1

    def longprune(self, tcell: np.ndarray, fix: np.ndarray):
        bottom = 0
        while bottom < self.fmptr:
            is_subset = True
            for i in range(self.n):
                if fix[i] & ~self.workspace[bottom + i]:
                    is_subset = False
                    break
            bottom += self.n
            if is_subset:
                tcell = np.bitwise_and(tcell, self.workspace[bottom:bottom + self.n])
            bottom += self.n
        return tcell

    def shortprune(self, set1: np.ndarray, set2: np.ndarray):
        return np.bitwise_and(set1, set2)

    def breakout(self, lab: np.ndarray, ptn: np.ndarray, level: int, tc: int, tv: int, active: np.ndarray):
        active[:] = False
        active[tc] = True

        i = tc
        prev = tv

        while True:
            next_val = lab[i]
            lab[i] = prev
            i += 1
            prev = next_val
            if prev == tv:
                break

        ptn[tc] = level

    def maketargetcell(self, g: np.ndarray, lab: np.ndarray, ptn: np.ndarray, level: int, tcell: np.ndarray,
                       tc_level: int, hint: int) -> (int, int):
        i = self.targetcell(g, lab, ptn, level, tc_level, hint)

        j = i + 1
        while ptn[j] > level:
            j += 1

        # Empty the tcell
        tcell[:] = False

        # Add elements to tcell
        for k in range(i, j + 1):
            tcell[lab[k]] = True

        return j - i + 1, i

    def targetcell(self, g: np.ndarray, lab: np.ndarray, ptn: np.ndarray, level: int,
                   tc_level: int, hint: int) -> int:
        if hint >= 0 and ptn[hint] > level and (hint == 0 or ptn[hint - 1] <= level):
            return hint
        elif level <= tc_level:
            return bestcell(self.n, g, lab, ptn, level)
        else:
            for i in range(self.n):
                if ptn[i] > level:
                    return i
            return 0
