# -*- coding: utf-8 -*-
# pylint: disable-msg=E0602, E1103
"""
This package contains all the abstraction patterns.
"""
from .rhythm import (
    RHYTHMSTART_PATTERN,
    SINUS_PATTERN,
    TACHYCARDIA_PATTERN,
    EXTRASYSTOLE_PATTERN,
    BRADYCARDIA_PATTERN,
    RHYTHMBLOCK_PATTERN,
    VFLUTTER_PATTERN,
    ASYSTOLE_PATTERN,
    BIGEMINY_PATTERN,
    TRIGEMINY_PATTERN,
    COUPLET_PATTERN,
    AFIB_PATTERN,
)
from .sig_deviation import generate_Deflection_Patterns, RDEFLECTION_PATTERN
from .segmentation.QRS import QRS_PATTERN
from .segmentation.pwave import PWAVE_PATTERN
from .segmentation.twave import TWAVE_PATTERN
from .segmentation.beats import FIRST_BEAT_PATTERN, CARDIAC_CYCLE_PATTERN
from ..observables import *
from kardioml.segmentation.teijeiro.model import Interval, ConstraintNetwork, Variable
from kardioml.segmentation.teijeiro.model.automata import ABSTRACTED
import numpy

###########################################################################
###Knowledge base, defined as a preordered list of abstraction patterns ###
###########################################################################

# Knowledge base for rhythm interpretation
RHYTHM_KNOWLEDGE = generate_Deflection_Patterns(2) + [
    RDEFLECTION_PATTERN,
    QRS_PATTERN,
    TWAVE_PATTERN,
    PWAVE_PATTERN,
    RHYTHMSTART_PATTERN,
    SINUS_PATTERN,
    TACHYCARDIA_PATTERN,
    TRIGEMINY_PATTERN,
    EXTRASYSTOLE_PATTERN,
    COUPLET_PATTERN,
    BIGEMINY_PATTERN,
    RHYTHMBLOCK_PATTERN,
    BRADYCARDIA_PATTERN,
    VFLUTTER_PATTERN,
    AFIB_PATTERN,
    ASYSTOLE_PATTERN,
]

# Knowledge base for wave segmentation
SEGMENTATION_KNOWLEDGE = generate_Deflection_Patterns(2) + [
    QRS_PATTERN,
    TWAVE_PATTERN,
    PWAVE_PATTERN,
    FIRST_BEAT_PATTERN,
    CARDIAC_CYCLE_PATTERN,
]

# Knowledge base for atrial fibrillation detection.
# KNOWLEDGE =  generate_Deflection_Patterns(2)+[RDEFLECTION_PATTERN, QRS_PATTERN,
#            TWAVE_PATTERN, RHYTHMSTART_PATTERN, AFIB_PATTERN, ASYSTOLE_PATTERN]

# Knowledge base for QRS delineation validation.
# KNOWLEDGE = [QRS_PATTERN]

# Knowledge base for beat interpretations (QT database validation)
# KNOWLEDGE =  generate_Deflection_Patterns(2) + [
#             QRS_PATTERN, TWAVE_PATTERN, PWAVE_PATTERN, SINUS_BEAT_PATTERN]


def set_knowledge_base(knowledge):
    """
    Sets the knowledge base to be used in the interpretation, and updates
    the necessary global variables.
    """
    global KNOWLEDGE, _OBSERVABLES, _ABDUCIBLES, _LMAP, _EXCLUSION
    KNOWLEDGE = knowledge
    # First, we check the consistency of every single abstraction pattern
    for p in KNOWLEDGE:
        # In the Environment and Abstracted sets no repeated types can happen.
        for qset in (p.abstracted, p.environment):
            for q in qset:
                # The only coincidence must be q
                assert len(set(q.mro()) & qset) == 1
        # There should be no subclass relations between hyp. and abstractions
        for q in p.abstracted:
            assert not p.Hypothesis in q.mro()
        assert not set(p.Hypothesis.mro()) & p.abstracted
        # The abstraction transitions must be properly set.
        for q in p.abstractions:
            assert q in p.abstracted
            for tr in p.abstractions[q]:
                assert tr.observable is q
                assert tr.abstracted is ABSTRACTED
                assert tr in p.transitions
    # Organization of all the observables in abstraction levels.
    _OBSERVABLES = set.union(*(({p.Hypothesis} | p.abstracted | p.environment) for p in KNOWLEDGE))
    _ABDUCIBLES = tuple(set.union(*(p.abstracted for p in KNOWLEDGE)))
    # To perform the level assignment, we use a constraint network.
    _CNET = ConstraintNetwork()
    # Mapping from observable types to abstraction levels.
    _LMAP = {}
    for q in _OBSERVABLES:
        _LMAP[q] = Variable(value=Interval(0, numpy.inf))
    # We set the restrictions, and minimize the network
    # All subclasses must have the same level than the superclasses
    for q in _OBSERVABLES:
        for sup in (set(q.mro()) & _OBSERVABLES) - {q}:
            _CNET.set_equal(_LMAP[q], _LMAP[sup])
    # Abstractions force a level increasing
    for p in KNOWLEDGE:
        for qabs in p.abstracted:
            _CNET.add_constraint(_LMAP[qabs], _LMAP[p.Hypothesis], Interval(1, numpy.inf))
    # Network minimization
    _CNET.minimize_network()
    # Now we assign to each observable the minimum of the solutions interval.
    for q in _OBSERVABLES:
        _LMAP[q] = int(_LMAP[q].start)
    # Manual definition of the exclusion relation between observables.
    _EXCLUSION = {
        Deflection: (Deflection,),
        QRS: (QRS,),
        TWave: (TWave,),
        PWave: (PWave,),
        CardiacCycle: (CardiacCycle,),
        Cardiac_Rhythm: (Cardiac_Rhythm,),
    }
    # Automatic expansion of the exclusion relation.
    for q, qexc in _EXCLUSION.iteritems():
        _EXCLUSION[q] = tuple((q2 for q2 in _OBSERVABLES if issubclass(q2, qexc)))
    for q in sorted(_OBSERVABLES, key=_LMAP.get, reverse=True):
        _EXCLUSION[q] = tuple(set.union(*(set(_EXCLUSION[q2]) for q2 in _EXCLUSION if issubclass(q, q2))))


def get_obs_level(observable):
    """Obtains the abstraction level of an observable"""
    return _LMAP[observable]


def is_abducible(observable):
    """Checks if an observable is abducible"""
    return issubclass(observable, _ABDUCIBLES)


def get_excluded(observable):
    """Obtains the tuple of observables excluded by a given one"""
    return _EXCLUSION[observable]


def get_max_level():
    """Obtains the maximum abstraction level of the domain."""
    return max(_LMAP.itervalues())


def get_level(nlevel, subclasses=True):
    """
    Obtains the set of observables of a certain abstraction level.

    Parameters
    ----------
    nlevel:
        Level of which we want the observables.
    subclasses:
        Boolean flag indicating if we want to explicitly return subclasses of
        the observables that may belong also to level n.
    """
    value = set(q for q in _LMAP if _LMAP[q] == nlevel)
    if not subclasses:
        value = set.union(*(qset for qset in (set(q.mro()) & value for q in value) if len(qset) == 1))
    return value


# By default, we select the full knowledge base
set_knowledge_base(RHYTHM_KNOWLEDGE)

if __name__ == "__main__":
    pass
