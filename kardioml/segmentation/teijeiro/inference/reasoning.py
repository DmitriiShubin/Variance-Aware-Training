"""
Created on Mon Jun 11 18:27:22 2012

This module contains the definition of the reasoning methods using to generate
the successor interpretations of a given one. Several different methods are
applied based on the internal state of the interpretation, but the result is
equivalent to apply always the abduction process while avoiding the generation
of many uninteresting interpretations.

@author: T. Teijeiro
"""
import kardioml.segmentation.teijeiro.knowledge.abstraction_patterns as ap
import kardioml.segmentation.teijeiro.knowledge.constants as C
from kardioml.segmentation.teijeiro.model.observable import is_singleton
from kardioml.segmentation.teijeiro.model.FreezableObject import clone_attrs
from kardioml.segmentation.teijeiro.model import AbstractionPattern
from kardioml.segmentation.teijeiro.model.interpretation import Interpretation
from kardioml.segmentation.teijeiro.model.automata import NULL_PROC
from collections import Counter
import copy
import itertools as it
import numpy as np
import weakref
import sortedcontainers
from ..model.constraint_network import InconsistencyError
import kardioml.segmentation.teijeiro.knowledge.observables as OBS
import kardioml.segmentation.teijeiro.acquisition.obs_buffer as BUF


# numpy.any behavior with generators is not as expected, so we require to modify
# it
if any is np.any:
    import __builtin__

    any = __builtin__.any
assert any is not np.any


#########################
### Module attributes ###
#########################

# Flag to avoid dead-end branch pruning, to allow debugging of all hypotheses
SAVE_TREE = False

# Flag to avoid the merging strategy exploration.
MERGE_STRATEGY = True

# Counter to measure where the interpretation process concentrates its efforts.
STATS = Counter()

# Sorted list of interpretations to find merge situations.
_INCACHE = sortedcontainers.SortedSet(
    key=lambda i: (len(i.observations), len(i.focus), len(i.unintelligible), len(i.abstracted))
)

# Dictionary to map merged interpretations
_MERGED = weakref.WeakKeyDictionary()

# Dictionary to obtain the successor generator of each interpretation.
_SUCCESSORS = weakref.WeakKeyDictionary()

# Set containing the interpretations with one or more firm successors. If an
# interpretation is exhausted and does not belong to this set, it is discarded
# as a dead-end reasoning path.
_FSUCC = weakref.WeakSet()

# Set containing the findings that have been successfully matched after a
# predictive reasoning
_MATCHED_FINDINGS = weakref.WeakSet()

###############################
### Global module functions ###
###############################


def reset():
    """
    Deletes the content of all the global data structures of the module,
    allowing a fresh restart of every reasoning operations.
    """
    global _SUCCESSORS
    _INCACHE.clear()
    _MERGED.clear()
    # We cannot clear this dict, it raises many KeyError exceptions.
    _SUCCESSORS = weakref.WeakKeyDictionary()
    _FSUCC.clear()
    _MATCHED_FINDINGS.clear()


def clear_cache(time):
    """
    Deletes old information from interpretation cache. All interpretations
    with time_point < time are removed, and therefore cannot be eligible for
    merging.
    """
    global _INCACHE
    _INCACHE = sortedcontainers.SortedSet(
        (i for i in _INCACHE if i.time_point >= time),
        key=lambda i: (len(i.observations), len(i.focus), len(i.unintelligible), len(i.abstracted)),
    )


#########################
### Utility functions ###
#########################


def save_hierarchy(interpretation, sset):
    """
    This function adds all the hierarchy of a given *interpretation* in *sset*.
    """
    node = interpretation
    while node is not None:
        # When an interpretation is already found in the set, we can stop.
        if node in sset:
            break
        sset.add(node)
        node = node.parent


def _singleton_violation(pat, interpretation):
    """
    Checks if an abstraction pattern violates the singleton constraints in a
    given interpretation, that is, if the interpretation already contains an
    observation of the same type of the hypothesis of the pattern, and that
    hypothesis is a singleton.

    Parameters
    ----------
    pat:
        PatternAutomata.
    interpretation:
        Interpretation.
    """
    return issubclass(pat.Hypothesis, tuple(interpretation.singletons))


def _find_mergeable(interpretation):
    """
    Searches the interpretation cache in order to find mergeable
    interpretations.
    """
    if not MERGE_STRATEGY:
        return None
    idx = _INCACHE.bisect_left(interpretation)
    nobs = len(interpretation.observations)
    nfoc = len(interpretation.focus)
    nunint = len(interpretation.unintelligible)
    nabs = len(interpretation.abstracted)
    mergeable = interpretation.is_mergeable
    while idx < len(_INCACHE):
        other = _INCACHE[idx]
        if (
            len(other.observations) > nobs
            or len(other.focus) != nfoc
            or len(other.unintelligible) != nunint
            or len(other.abstracted) != nabs
        ):
            break
        if mergeable(other):
            return other
        idx += 1
    return None


def _finding_match(interp, finding, obs, start, end, pred, succ, is_abstr):
    """
    Performs a matching operation between a finding and an observation in a
    given interpretation, checking some constraints set by some additional
    parameters.

    Parameters
    ----------
    interp:
        Interpretation in which the matching is performed.
    finding:
        Finding to match.
    obs:
        Matched observation.
    start:
        Previous start time of the hypothesis of the pattern that generated
        *finding*.
    end:
        Previous end time of the hypothesis of the pattern that generated
        *finding*.
    pred:
        Predecessor of *finding* according to the consecutivity constraints.
    succ:
        Sucessor of *finding* according to the consecutivity constraints.
    is_abstr:
        Flag indicating if *finding* is abstracted.
    """
    interp.focus.match(finding, obs)
    newhyp = interp.focus.top[0]
    # Exclusion and consecutivity violations of the hypothesis are
    # checked only if the value changes.
    if newhyp.end.value != end or newhyp.start.value != start:
        interp.verify_exclusion(newhyp)
        interp.verify_consecutivity_violation(newhyp)
    _MATCHED_FINDINGS.add(finding)
    if is_abstr:
        # Lazy copy of the parent reference
        interp.abstracted = interp.abstracted.copy()
        interp.abstracted.add(obs)
    if pred is not None or succ is not None:
        interp.predinfo = interp.predinfo.copy()
        if pred is not None:
            interp.predinfo[obs] = (pred, type(finding))
        if succ is not None:
            interp.predinfo[succ] = (obs, type(finding))


###########################
### Reasoning functions ###
###########################


def firm_succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one, but it
    only yields those that are firm. To do this, it performs a depth-first
    search process.

    **Note:** This generator will not obtain **every** node in the tree, but
    only those that are firm. Thus, a walk using only the output of this
    generator will not be exhaustive. Completeness is only guaranteed by
    :func:`multicall_succ`.
    """
    # Check merge possibilities
    if interpretation not in _SUCCESSORS:
        mrg = _MERGED.get(interpretation) or _find_mergeable(interpretation)
        if mrg is not None:
            _SUCCESSORS[interpretation] = _merge_succ(interpretation, mrg)
            _MERGED[interpretation] = mrg
    stack = [(interpretation, multicall_succ(interpretation))]
    while stack:
        node, generator = stack[-1]
        finished = False
        while not finished:
            try:
                suc = generator.next()
                if suc.is_firm:
                    # Only original interpretations are cached.
                    if MERGE_STRATEGY and interpretation not in _MERGED:
                        _INCACHE.add(suc)
                    # FIXME ad-hoc code to improveme sinus rhythm valuations
                    if suc.focus and isinstance(suc.focus.top[0], OBS.Sinus_Rhythm):
                        hyp, pat = suc.focus.top
                        llim, ulim = (
                            pat.evidence[OBS.QRS][-2].time.start,
                            pat.evidence[OBS.QRS][-1].time.start,
                        )
                        abst = set(
                            BUF.get_observations(
                                start=llim, end=ulim, filt=lambda d: d not in suc.abstracted
                            )
                        )
                        if abst:
                            suc.abstracted = suc.abstracted.copy()
                            suc.abstracted.update(abst)
                    # END of ad-hoc code
                    save_hierarchy(node, _FSUCC)
                    yield suc
                else:
                    stack.append((suc, multicall_succ(suc)))
                    finished = True
            except StopIteration:
                finished = True
                # Remove interpretations with no firm descendants
                if not SAVE_TREE and node not in _FSUCC and not node.is_firm:
                    node.discard('Dead-end interpretation')
                stack.pop()


def multicall_succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one. The
    sequence is the same that using the **_succ** generator, but with the
    difference that it allows the creation and iteration of many generators
    ensuring each successor interpretation is created only once.
    """
    # Generator lookup or creation.
    successors = _SUCCESSORS.get(interpretation)
    if successors is None:
        merged = _MERGED.get(interpretation)
        successors = _succ(interpretation) if merged is None else _merge_succ(interpretation, merged)
        _SUCCESSORS[interpretation] = successors
    # Main loop
    yielded = weakref.WeakSet()
    while True:
        nxt = next((n for n in interpretation.child if n not in yielded), None)
        nxt = nxt or successors.next()
        yielded.add(nxt)
        yield nxt


def _succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one, by the
    appropriate combination of the different implemented reasoning modes.
    **Warning**: This function can be called only one time for each
    interpretation. If you need multiple iterations over the successor nodes
    of an interpretation, use the *multicall_succ* generator.
    """
    sequences = []
    focus, pat = interpretation.focus.top
    if pat is not None and focus is pat.finding:
        sequences.append(subsume(interpretation, focus, pat))
        # Past environment findings are not predicted.
        if focus.lateend > interpretation.time_point or pat.abstracts(focus):
            sequences.append(predict(interpretation, focus, pat))
    else:
        if pat is not None:
            sequences.append(deduce(interpretation, focus, pat))
        sequences.append(abduce(interpretation, focus, pat))
        sequences.append(advance(interpretation, focus, pat))
    return it.chain.from_iterable(sequences)


def _merge_succ(interpretation, merged):
    """
    Returns a generator of the successor interpretations of a given one,
    assuming we have performed a merge operation with an existing
    interpretation.
    """
    diff = interpretation.past_metrics.diff(merged.past_metrics)
    nabdiff = interpretation.nabd - merged.nabd
    for succ in multicall_succ(merged):
        # Successors are shallow copies of the merged successors, changing
        # the tree structure references and the past_metrics.
        nxt = copy.copy(succ)
        nxt.name = str(Interpretation.counter)
        Interpretation.counter += 1
        nxt.parent = interpretation
        nxt.child = []
        nxt.past_metrics = nxt.past_metrics.patch(diff)
        nxt.nabd += nabdiff
        # We already flag the successors as merged.
        _MERGED[nxt] = succ
        STATS.update(['Merge'])
        yield nxt


def subsume(interp, finding, pattern):
    """
    Obtains the interpretations that can be derived from a given one, by
    establishing all the consistent matchings with the unmatched finding of
    the interpretation through a subsumption operation.
    """
    is_abstr = pattern.abstracts(finding)
    start, end = pattern.hypothesis.start.value, pattern.hypothesis.end.value
    pred, succ = pattern.get_consecutive(finding)

    def valid_subsumption(ev):
        """Constraints that can be checked before subsumption"""
        return (
            ev.earlystart in finding.start.value
            and ev.time.start in finding.time.value
            and ev.lateend in finding.end.value
            and ev not in interp.unintelligible
            and (ev not in interp.abstracted if is_abstr else True)
        )

    opt = []
    if pred is not None:
        subs = next(
            interp.get_observations(
                clazz=type(finding), start=pred.earlyend, end=finding.lateend, filt=valid_subsumption
            ),
            None,
        )
        if subs is not None:
            opt = [subs]
    elif succ is not None:
        if succ:
            if succ in interp.predinfo:
                subs = interp.predinfo[succ][0]
                if valid_subsumption(subs):
                    opt = [subs]
            else:
                subs = next(
                    interp.get_observations(
                        clazz=type(finding),
                        start=finding.earlystart,
                        end=succ.latestart,
                        filt=valid_subsumption,
                        reverse=True,
                    ),
                    None,
                )
            if subs is not None:
                opt = [subs]
    else:
        opt = interp.get_observations(
            clazz=type(finding), start=finding.earlystart, end=finding.lateend, filt=valid_subsumption
        )
    for subs in opt:
        newint = Interpretation(interp)
        try:
            _finding_match(newint, finding, subs, start, end, pred, succ, is_abstr)
            newint.remove_old()
            STATS.update(['S+' + str(finding.__class__.__name__)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))


def predict(interp, finding, pattern):
    """
    Obtains the interpretations that can be derived from a given one, by
    performing a deduction operation taking as hypothesis the unmatched finding
    of the interpretation.
    """
    for pat in (
        p
        for p in ap.KNOWLEDGE
        if issubclass(p.Hypothesis, type(finding)) and not _singleton_violation(p, interp)
    ):
        # The exploration stops at the first consistent matching of the finding
        if finding in _MATCHED_FINDINGS:
            _MATCHED_FINDINGS.remove(finding)
            return
        newint = Interpretation(interp)
        if is_singleton(pat.Hypothesis):
            newint.singletons = newint.singletons.copy()
            newint.singletons.add(pat.Hypothesis)
        pattern = AbstractionPattern(pat)
        # We set the known attributes of the new conjecture
        clone_attrs(pattern.hypothesis, finding)
        try:
            # The new predicted observation is focused.
            newint.focus.push(pattern.hypothesis, pattern)
            newint.verify_consecutivity_violation(pattern.hypothesis)
            newint.verify_exclusion(pattern.hypothesis)
            # And the number of abducible observations is increased.
            if ap.is_abducible(pat.Hypothesis):
                newint.nabd += 1
            STATS.update(['D+' + str(pat)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))


def deduce(interp, focus, pattern):
    """
    Extends the pattern whose hypothesis is the current inference focus of the
    interpretation, by testing all possible successor paths.
    """
    assert pattern.finding is None
    for suc in pattern.successors():
        # Once a sufficient set of evidence has ben obtained for this pattern
        # and all dependent patterns up in the abstraction hierarchy, we don't
        # look for other alternatives. Therefore, the successors() generator
        # should try to optimize the coverage of the generated patterns.
        if interp in _FSUCC:
            return
        newint = Interpretation(interp)
        try:
            newint.focus.top = (suc.hypothesis, suc)
            # We set the focus on the new predicted finding, if it exists.
            if suc.finding is not None:
                bef, aft = suc.get_consecutive(suc.finding)
                if bef is not None:
                    newint.verify_consecutivity_satisfaction(bef, suc.finding, type(suc.finding))
                elif aft is not None:
                    newint.verify_consecutivity_satisfaction(suc.finding, aft, type(suc.finding))
                newint.focus.push(suc.finding, suc)
            # If the hypothesis timing changes, consecutivity and exclusion
            # constraints have to be checked.
            if (
                suc.hypothesis.end.value != focus.end.value
                or suc.hypothesis.start.value != focus.start.value
            ):
                newint.verify_exclusion(suc.hypothesis)
                newint.verify_consecutivity_violation(suc.hypothesis)
            STATS.update(['X+' + str(pattern.automata)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))


def abduce(interp, focus, pattern):
    """
    Continues the inference by performing an abduction operation on the current
    inference focus of the interpretation.
    """
    # If the focus is an hypothesis of a pattern, we require that pattern to
    # have sufficient supporting evidence.
    if (
        (pattern is not None and not pattern.sufficient_evidence)
        or focus in interp.abstracted
        or interp.focus.get_delayed_finding(focus) is not None
    ):
        return
    if pattern is not None and pattern.automata.obs_proc is not NULL_PROC:
        pattern = copy.copy(pattern)
        try:
            pattern.finish()
            focus = pattern.hypothesis
        except InconsistencyError as error:
            return
    qobs = type(focus)
    for pat in (
        p
        for p in ap.KNOWLEDGE
        if issubclass(qobs, tuple(p.abstracted)) and not _singleton_violation(p, interp)
    ):
        types = [tp for tp in pat.abstracted if issubclass(qobs, tp)]
        for typ in types:
            for trans in pat.abstractions[typ]:
                try:
                    newint = Interpretation(interp)
                    if pattern is not None:
                        newint.observations = newint.observations.copy()
                        newint.observations.add(focus)
                    # Pattern consistency checking
                    pattern = AbstractionPattern(pat)
                    pattern.istate = trans.istate
                    pattern.fstate = trans.fstate
                    pattern.trseq.append((trans, focus))
                    pattern.evidence[typ].append(focus)
                    trans.tconst(pattern, focus)
                    pattern.check_temporal_consistency()
                    trans.gconst(pattern, focus)
                    # Interpretation updating
                    newint.abstracted = newint.abstracted.copy()
                    newint.abstracted.add(focus)
                    if is_singleton(pat.Hypothesis):
                        newint.singletons = newint.singletons.copy()
                        newint.singletons.add(pat.Hypothesis)
                    if ap.is_abducible(pat.Hypothesis):
                        newint.nabd += 1
                    newint.focus.pop()
                    newint.focus.push(pattern.hypothesis, pattern)
                    newint.verify_exclusion(pattern.hypothesis)
                    newint.verify_consecutivity_violation(pattern.hypothesis)
                    STATS.update(['A+' + str(pat)])
                    yield newint
                except InconsistencyError as error:
                    newint.discard(str(error))


def advance(interp, focus, pattern):
    """
    Continues the inference by recovering the previous focus of attention, or
    by going to the next unexplained observation. It also resolves all the
    postponed matchings.
    """
    max_ap_level = ap.get_max_level()
    # We can not advance if the focus is an hypothesis of a pattern with
    # insufficient evidence.
    newint = None
    if pattern is not None:
        if not pattern.sufficient_evidence:
            return
        # If there is an observation procedure for the focused hypothesis, the
        # execution takes place now.
        elif pattern.automata.obs_proc is not NULL_PROC:
            patcp = copy.copy(pattern)
            try:
                patcp.finish()
                newint = Interpretation(interp)
                if (
                    focus.end.value != patcp.hypothesis.end.value
                    or focus.start.value != patcp.hypothesis.start.value
                ):
                    newint.verify_consecutivity_violation(patcp.hypothesis)
                    newint.verify_exclusion(patcp.hypothesis)
                focus = patcp.hypothesis
            except InconsistencyError:
                return
    # If the new focus is a delayed matching, we solve it at this point.
    finding = interp.focus.get_delayed_finding(interp.focus.top[0])
    if finding is not None:
        newint = newint or Interpretation(interp)
        try:
            newint.focus.pop()
            pattern = newint.focus.top[1]
            pred, succ = pattern.get_consecutive(finding)
            _finding_match(
                newint,
                finding,
                focus,
                pattern.hypothesis.start.value,
                pattern.hypothesis.end.value,
                pred,
                succ,
                pattern.abstracts(finding),
            )
            # The matched hypothesis is included in the observations list
            newint.observations = newint.observations.copy()
            newint.observations.add(focus)
        except InconsistencyError as error:
            newint.discard(str(error))
            return
    else:
        # HINT with the current knowledge base, we restrict the advancement to
        # observations of the first or the last level, not allowing partial
        # interpretations in the abstraction level.
        if len(interp.focus) == 1 and 0 < ap.get_obs_level(type(focus)) < max_ap_level:
            return
        # We just move on the focus.
        newint = newint or Interpretation(interp)
        if ap.get_obs_level(type(focus)) == 0:
            newint.unintelligible = newint.unintelligible.copy()
            newint.unintelligible.add(focus)
        # If the focus is a hypothesis, it is added to the observations list
        if pattern is not None:
            newint.observations = newint.observations.copy()
            newint.observations.add(focus)
        newint.focus.pop()
    # If we have reached the top of the stack, we go ahead to the next
    # unexplained observation.
    if not newint.focus:
        try:
            unexp = newint.get_observations(
                start=focus.earlystart + 1,
                filt=lambda ev: ev not in newint.unintelligible
                and ev not in newint.abstracted
                and ap.is_abducible(type(ev)),
            ).next()
            newint.focus.push(unexp, None)
        except StopIteration:
            newint.discard('No unexplained evidence after the current point')
            return
    # Finally, we remove old observations from the interpretation, since they
    # won't be used in following reasoning steps, and they affect the
    # performance of the searching procedure.
    oldtime = max(newint.past_metrics.time, newint.focus.earliest_time) - C.FORGET_TIMESPAN
    newint.remove_old(oldtime)
    yield newint


if __name__ == "__main__":
    pass
