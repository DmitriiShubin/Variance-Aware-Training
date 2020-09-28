# -*- coding: utf-8 -*-
# pylint: disable-msg=E1103
"""
Created on Fri Sep 20 16:10:22 2013

This module contains the abstraction pattern definitions of regular cardiac
rhythms.

@author: T. Teijeiro
"""

import kardioml.segmentation.teijeiro.knowledge.observables as o
import kardioml.segmentation.teijeiro.acquisition.signal_buffer as sig_buf
from kardioml.segmentation.teijeiro.knowledge.abstraction_patterns.segmentation.QRS import QRS_SHAPES
import kardioml.segmentation.teijeiro.knowledge.constants as C
from kardioml.segmentation.teijeiro.model import ConstraintNetwork, verify, Interval as Iv
from kardioml.segmentation.teijeiro.model.automata import (
    PatternAutomata,
    ABSTRACTED,
    ENVIRONMENT,
    BASIC_TCONST,
)
from kardioml.segmentation.teijeiro.utils.signal_processing.xcorr_similarity import (
    xcorr_valid,
    signal_match,
    signal_unmatch,
)
from kardioml.segmentation.teijeiro.utils.units_helper import msec2samples as ms2sp, samples2sec as sp2sc
from collections import Counter
import numpy as np
import copy

#####################################################
### New definition of the regular rhythm patterns ###
#####################################################


def _update_morphology(pattern):
    """
    Updates the reference morphology of the hypothesis of the pattern from
    the morphology of the beats that are part of the rhythm.
    """
    beats = pattern.evidence[o.QRS]
    for lead in sig_buf.get_available_leads():
        # We get the most common pattern as the reference.
        ctr = Counter(b.shape[lead].tag for b in beats if lead in b.shape)
        if ctr:
            mc = ctr.most_common(2)
            # If the most common is not unique, we move on
            if len(mc) == 2 and mc[0][1] == mc[1][1]:
                continue
            tag = mc[0][0]
            energy = np.mean(
                [b.shape[lead].energy for b in beats if lead in b.shape and b.shape[lead].tag == tag]
            )
            if not lead in pattern.hypothesis.morph:
                pattern.hypothesis.morph[lead] = o.QRSShape()
            pattern.hypothesis.morph[lead].tag = tag
            pattern.hypothesis.morph[lead].energy = energy


def _update_measures(pattern):
    """
    Updates the cycle time measures of the pattern.
    """
    # Maximum number of observations considered for the measures (to avoid
    # excessive influence of old observations)
    nobs = 30
    beats = pattern.evidence[o.QRS][-nobs:]
    obseq = pattern.obs_seq
    # RR
    rrs = np.diff([b.time.start for b in beats])
    # The RT (QT) measure is updated by a Kalman Filter strategy.
    # Belief values
    rtmean, rtstd = pattern.hypothesis.meas.rt
    # Current RR measure (bounded)
    qrs = beats[-1]
    rr = rrs[-1]
    rr = max(min(rr, C.QTC_RR_LIMITS.end), C.QTC_RR_LIMITS.start)
    # Kalman filter algorithm, as explained in "Probabilistic Robotics"
    sigma_tbar = rtstd ** 2 + C.KF_Q ** 2
    twave = obseq[-1]
    if isinstance(twave, o.TWave):
        # rt and corrected rt measure in the current iteration
        rt = twave.earlyend - qrs.time.start
        rtc = ms2sp(1000.0 * sp2sc(rt) / np.cbrt(sp2sc(rr)))
        meas_err = rtc - rtmean
        # Abnormally QT intervals have associated higher uncertainty
        qt = twave.earlyend - qrs.earlystart
        qt_lims = C.QT_FROM_RR(Iv(rr, rr))
        # Measure uncertainty, represented by the R matrix in the Kalman filter
        KF_R = meas_err if qt in qt_lims else ms2sp(120)
        k_t = sigma_tbar / (sigma_tbar + max(KF_R, C.MIN_QT_STD) ** 2)
    else:
        # No measure - 0 Kalman gain
        meas_err = 0
        k_t = 0
    if rtmean == 0:
        mu_t = meas_err
        sigma_t = C.QT_ERR_STD ** 2
    else:
        mu_t = rtmean + k_t * meas_err
        sigma_t = (1.0 - k_t) * sigma_tbar
    # PQ
    pqs = []
    for pwave in pattern.evidence[o.PWave][-nobs:]:
        i = pattern.get_step(pwave)
        qrs = obseq[i - 1]
        pqs.append(qrs.earlystart - pwave.earlystart)
    pattern.hypothesis.meas = o.CycleMeasurements(
        (np.mean(rrs), np.std(rrs)), (mu_t, np.sqrt(sigma_t)), (np.mean(pqs), np.std(pqs))
    )


def _check_missed_beats(pattern):
    """
    Checks if a rhythm pattern has missed a QRS complex in the identification,
    by looking for a waveform "identical" to the last observed in the interval
    between the last two observations.
    """
    qrs = pattern.evidence[o.QRS][-1]
    obseq = pattern.obs_seq
    idx = obseq.index(qrs)
    if idx > 0:
        prevobs = next((obs for obs in reversed(obseq[:idx]) if obs is not None), None)
        if prevobs is not None:
            if isinstance(prevobs, o.QRS):
                limit = max(prevobs.lateend, prevobs.earlystart + qrs.lateend - qrs.earlystart)
            else:
                limit = prevobs.lateend
        else:
            limit = pattern.hypothesis.earlystart
        ulimit = qrs.earlystart - C.TACHY_RR.start
        if limit >= ulimit:
            return
        sig = {}
        # We take the signal fragment with maximum correlation with the QRS
        # signal in each lead, and we check if the two fragments can be
        # clustered as equal QRS complexes.
        qshape = {}
        corr = -np.Inf
        delay = 0
        leads = sig_buf.get_available_leads()
        for lead in leads:
            qshape[lead] = o.QRSShape()
            sigfr = sig_buf.get_signal_fragment(qrs.earlystart, qrs.lateend + 1, lead=lead)[0]
            qshape[lead].sig = sigfr - sigfr[0]
            qshape[lead].amplitude = np.ptp(qshape[lead].sig)
            sig[lead] = sig_buf.get_signal_fragment(limit, ulimit, lead=lead)[0]
            lcorr, ldelay = xcorr_valid(sig[lead], qshape[lead].sig)
            if lcorr > corr:
                corr, delay = lcorr, ldelay
        if 0 <= delay < len(sig[lead]):
            sshape = {}
            for lead in leads:
                sshape[lead] = o.QRSShape()
                sshape[lead].sig = sig[lead][delay : delay + len(qshape[lead].sig)] - sig[lead][delay]
                sshape[lead].amplitude = np.ptp(sshape[lead].sig)
            if isinstance(pattern.hypothesis, o.RegularCardiacRhythm):
                qref = pattern.evidence[o.QRS][-2]
                rr = float(qrs.earlystart - qref.earlystart)
                loc = (limit + delay - qref.earlystart) / rr
                # Check for one and two missed beats in regular positions
                if 0.45 <= loc <= 0.55:
                    verify(signal_unmatch(sshape, qshape), 'Missed beat')
                elif 0.28 <= loc <= 0.38 and not signal_unmatch(sshape, qshape):
                    corr = -np.Inf
                    delay = 0
                    for lead in leads:
                        sig[lead] = sig_buf.get_signal_fragment(
                            int(qref.earlystart + 0.61 * rr),
                            min(
                                int(qref.earlystart + 0.71 * rr) + len(qshape[lead].sig), int(qrs.earlystart)
                            ),
                            lead=lead,
                        )[0]
                        lcorr, ldelay = xcorr_valid(sig[lead], qshape[lead].sig)
                        if lcorr > corr:
                            corr, delay = lcorr, ldelay
                    sshape = {}
                    for lead in leads:
                        sshape[lead] = o.QRSShape()
                        sshape[lead].sig = (
                            sig[lead][delay : delay + len(qshape[lead].sig)] - sig[lead][delay]
                        )
                        sshape[lead].amplitude = np.ptp(sshape[lead].sig)
                    verify(signal_unmatch(sshape, qshape), 'Two missed beats')
                elif 0.61 <= loc <= 0.71 and not signal_unmatch(sshape, qshape):
                    corr = -np.Inf
                    delay = 0
                    for lead in leads:
                        sig[lead] = sig_buf.get_signal_fragment(
                            int(qref.earlystart + 0.28 * rr),
                            min(
                                int(qref.earlystart + 0.38 * rr) + len(qshape[lead].sig), int(qrs.earlystart)
                            ),
                            lead=lead,
                        )[0]
                        lcorr, ldelay = xcorr_valid(sig[lead], qshape[lead].sig)
                        if lcorr > corr:
                            corr, delay = lcorr, ldelay
                    sshape = {}
                    for lead in leads:
                        sshape[lead] = o.QRSShape()
                        sshape[lead].sig = (
                            sig[lead][delay : delay + len(qshape[lead].sig)] - sig[lead][delay]
                        )
                        sshape[lead].amplitude = np.ptp(sshape[lead].sig)
                    verify(signal_unmatch(sshape, qshape), 'Two missed beats')
            else:
                verify(signal_unmatch(sshape, qshape), 'Missed beat')


def _cycle_finished_gconst(pattern, _):
    """
    General constraints to be added when a new cycle is observed, which
    currently coincides with the observation of the T waves or a QRS complex
    not followed by an observed T wave.
    """
    # We update the measurements and the morphology of the rhythm.
    _update_measures(pattern)
    _update_morphology(pattern)
    # And check that there are no missed beat forms.
    _check_missed_beats(pattern)
    beats = pattern.evidence[o.QRS]
    rrs = np.diff([b.time.start for b in beats[-32:]])
    # HINT with this check, we avoid overlapping between sinus rhythms and
    # tachycardias and bradycardias at the beginning of the pattern.
    if len(beats) == 3:
        if pattern.automata is SINUS_PATTERN:
            verify(np.any(rrs < C.BRADY_RR.start))
        elif pattern.automata is BRADYCARDIA_PATTERN:
            if pattern.evidence[o.Cardiac_Rhythm] and isinstance(
                pattern.evidence[o.Cardiac_Rhythm][0], o.Sinus_Rhythm
            ):
                verify(any([rr not in C.SINUS_RR for rr in rrs]))
    elif len(beats) == 4:
        if pattern.automata is SINUS_PATTERN:
            verify(np.any(rrs > C.TACHY_RR.end))
        elif pattern.automata is TACHYCARDIA_PATTERN:
            if pattern.evidence[o.Cardiac_Rhythm] and isinstance(
                pattern.evidence[o.Cardiac_Rhythm][0], o.Sinus_Rhythm
            ):
                verify(any([rr not in C.SINUS_RR for rr in rrs]))
    # We impose some constraints in the evolution of the RR interval and
    # of the amplitude
    # TODO remove these lines to enable full check
    ######################################################################
    if len(beats) >= 3:
        # The coefficient of variation within a regular rhythm has to be low
        verify(np.std(rrs) / np.mean(rrs) <= C.RR_MAX_CV)
        # RR evolution
        meanrr, stdrr = pattern.hypothesis.meas.rr
        verify(
            meanrr - 2 * stdrr <= rrs[-1] <= meanrr + 2 * stdrr
            or abs(rrs[-1] - rrs[-2]) <= C.RR_MAX_DIFF
            or 0.8 * rrs[-2] <= rrs[-1] <= 1.2 * rrs[-2]
        )
    return
    #######################################################################
    # Morphology check. We require the rhythm morphology to be matched
    # by the new beat in the sequence.
    ref = pattern.hypothesis.morph
    # We initialize the morphology with the first beat.
    if not ref:
        for lead in beats[0].shape:
            ref[lead] = o.QRSShape()
            ref[lead].tag = beats[0].shape[lead].tag
            ref[lead].energy = beats[0].shape[lead].energy
    beat = beats[-1]
    # The leads matching morphology should sum more energy than the
    # unmatching.
    menerg = 0.0
    uenerg = 0.0
    perfect_match = False
    for lead in beat.shape:
        if lead in ref:
            bshape = beat.shape[lead]
            # If there is a "perfect" match in one lead, we accept clustering
            if bshape.tag == ref[lead].tag and 0.75 <= bshape.energy / ref[lead].energy <= 1.25:
                perfect_match = True
                break
            # If there are at least 10 beats in the sequence, we require
            # match from the beat to the rhythm, else we are ok in both
            # directions.
            match = bshape.tag in QRS_SHAPES[ref[lead].tag]
            if len(beats) < 10:
                match = bool(match or ref[lead].tag in QRS_SHAPES[bshape.tag])
            if match:
                menerg += ref[lead].energy
            else:
                uenerg += ref[lead].energy
    # If the matched energy is lower than unmatched, the hypothesis is
    # refuted.
    verify(perfect_match or menerg > uenerg)
    _update_morphology(pattern)
    if len(beats) >= 3:
        # RR evolution
        rr_prev = beats[-2].time.start - beats[-3].time.start
        rr_act = beats[-1].time.start - beats[-2].time.start
        verify(abs(rr_act - rr_prev) <= C.RR_MAX_DIFF)


def _p_qrs_tconst(pattern, pwave):
    """
    Temporal constraints of the P Waves wrt the corresponding QRS complex
    """
    BASIC_TCONST(pattern, pwave)
    tnet = pattern.last_tnet
    tnet.add_constraint(pwave.start, pwave.end, C.PW_DURATION)
    # We find the QRS observed just before that P wave.
    idx = pattern.get_step(pwave)
    if idx > 0 and isinstance(pattern.trseq[idx - 1][1], o.QRS):
        qrs = pattern.trseq[idx - 1][1]
        # PR interval
        tnet.add_constraint(pwave.start, qrs.start, C.N_PR_INTERVAL)
        tnet.set_before(pwave.end, qrs.start)
        if len(pattern.evidence[o.PWave]) > 10:
            # The mean and standard deviation of the PQ measurements will
            # influence the following observations.
            pqmean, pqstd = pattern.hypothesis.meas.pq
            interv = Iv(int(pqmean - 2 * pqstd), int(pqmean + 2 * pqstd))
            if interv.overlap(C.N_PR_INTERVAL):
                tnet.add_constraint(pwave.start, qrs.start, interv)


def _t_qrs_tconst(pattern, twave):
    """
    Temporal constraints of thw T Waves wrt the corresponding QRS complex.
    """
    BASIC_TCONST(pattern, twave)
    obseq = pattern.obs_seq
    idx = pattern.get_step(twave)
    try:
        tnet = pattern.last_tnet
        # We find the qrs observation precedent to this T wave.
        qrs = next(obseq[i] for i in range(idx - 1, -1, -1) if isinstance(obseq[i], o.QRS))
        # If we have more than one QRS, it is possible to constrain even more
        # the location of the T-Wave, based on rhythm information.
        qidx = pattern.evidence[o.QRS].index(qrs)
        if qidx > 0:
            refrr = qrs.time.end - pattern.evidence[o.QRS][qidx - 1].time.start
            refrr = max(min(refrr, C.QTC_RR_LIMITS.end), C.QTC_RR_LIMITS.start)
            rtc, rtstd = pattern.hypothesis.meas.rt
            if rtc > 0:
                # Expected QT value from the QT corrected value
                rtmean = ms2sp(1000.0 * sp2sc(rtc) * np.cbrt(sp2sc(refrr)))
                tnet.add_constraint(qrs.time, twave.end, Iv(rtmean - 2.5 * rtstd, rtmean + 2.5 * rtstd))
            tnet.add_constraint(qrs.time, twave.end, C.QT_FROM_RR(Iv(refrr, refrr)))
        else:
            # QT duration
            tnet.add_constraint(qrs.start, twave.end, C.N_QT_INTERVAL)
        if idx > 0 and isinstance(obseq[idx - 1], o.PWave):
            pwave = obseq[idx - 1]
            tnet.add_constraint(
                pwave.end, twave.start, Iv(C.ST_INTERVAL.start, C.PQ_INTERVAL.end + C.QRS_DUR.end)
            )
        # ST interval
        tnet.add_constraint(qrs.end, twave.start, C.ST_INTERVAL)
    except StopIteration:
        pass


def _prev_rhythm_tconst(pattern, rhythm):
    """Temporal constraints of a cardiac rhythm with the precedent one."""
    BASIC_TCONST(pattern, rhythm)
    tnet = pattern.last_tnet
    tnet.set_equal(pattern.hypothesis.start, rhythm.end)
    tnet.add_constraint(pattern.hypothesis.start, pattern.hypothesis.end, Iv(C.TACHY_RR.start, np.inf))


def _prev_rhythm_gconst(pattern, rhythm):
    """General constraints of a cardiac rhythm with the preceden one."""
    # We only accept the concatenation of the same rhythm for asystoles.
    verify(isinstance(pattern.hypothesis, o.Asystole) or type(pattern.hypothesis) != type(rhythm))
    # The rhythm measurements are initially taken from the previous rhythm.
    pattern.hypothesis.meas = o.CycleMeasurements(
        rhythm.meas.rr, (rhythm.meas.rt[0], C.QT_ERR_STD), (rhythm.meas.pq[0], C.QT_ERR_STD)
    )


def _rhythm_obs_proc(pattern):
    """Observation procedure executed once the rhythm pattern has finished"""
    # Tachycardias should have at least 4 QRS complexes.
    if isinstance(pattern.hypothesis, o.Tachycardia):
        verify(len(pattern.evidence[o.QRS]) >= 4)
    # We asign the endpoint of the hypothesis.
    pattern.hypothesis.end.value = pattern.evidence[o.QRS][-1].time.value


def _get_qrs_tconst(rr_bounds):
    """
    Obtains the temporal constraints definition function for regular rhythms
    with given RR limits.
    """
    # Temporal constraints definition on the QRS observations depend on the
    # rr_bounds parameter.
    def _qrs_tconst(pattern, qrs):
        """
        Temporal constraints to observe a new QRS complex.
        """
        beats = pattern.evidence[o.QRS]
        idx = beats.index(qrs)
        hyp = pattern.hypothesis
        tnet = pattern.last_tnet
        obseq = pattern.obs_seq
        oidx = pattern.get_step(qrs)
        # The environment complex sets the start of the rhythm observation.
        if pattern.get_evidence_type(qrs)[1] is ENVIRONMENT:
            tnet.set_equal(hyp.start, qrs.time)
        else:
            if idx > 0:
                prev = beats[idx - 1]
                tnet.remove_constraint(hyp.end, prev.time)
                # We create a new temporal network for the cyclic observations
                tnet = ConstraintNetwork()
                tnet.add_constraint(prev.time, qrs.time, rr_bounds)
                if rr_bounds is not C.TACHY_RR:
                    # Also bounding on begin and end, but with relaxed variation
                    # margin.
                    rlx_rrb = Iv(rr_bounds.start - C.TMARGIN, rr_bounds.end + C.TMARGIN)
                    tnet.add_constraint(prev.start, qrs.start, rlx_rrb)
                    tnet.add_constraint(prev.end, qrs.end, rlx_rrb)
                tnet.set_before(prev.end, qrs.start)
                # If there is a prior T Wave, it must finish before the start
                # of the QRS complex.
                if isinstance(obseq[oidx - 1], o.TWave):
                    prevt = obseq[oidx - 1]
                    tnet.set_before(prevt.end, qrs.start)
                ##RR evolution constraint. We combine the statistical limits
                # with a dynamic evolution.
                if idx > 1:
                    prev2 = beats[idx - 2]
                    rrev = prev.time.start - prev2.time.start
                    if hyp.meas.rr[0] > 0:
                        meanrr, stdrr = hyp.meas.rr
                        const = Iv(
                            min(0.8 * rrev, rrev - C.RR_MAX_DIFF, meanrr - 2 * stdrr),
                            max(1.2 * rrev, rrev + C.RR_MAX_DIFF, meanrr + 2 * stdrr),
                        )
                    else:
                        const = Iv(
                            min(0.8 * rrev, rrev - C.RR_MAX_DIFF), max(1.2 * rrev, rrev + C.RR_MAX_DIFF)
                        )
                    tnet.add_constraint(prev.time, qrs.time, const)
                pattern.temporal_constraints.append(tnet)
                # TODO improve
                if not qrs.frozen and hyp.morph:
                    nullsh = o.QRSShape()
                    refbeat = next(
                        (
                            b
                            for b in reversed(beats[:idx])
                            if not b.clustered
                            and all(
                                b.shape.get(lead, nullsh).tag == hyp.morph[lead].tag for lead in hyp.morph
                            )
                        ),
                        None,
                    )
                    if refbeat is not None:
                        qrs.shape = refbeat.shape
                        qrs.paced = refbeat.paced
        BASIC_TCONST(pattern, qrs)
        tnet.add_constraint(qrs.start, qrs.end, C.QRS_DUR)
        tnet.set_before(qrs.time, hyp.end)

    return _qrs_tconst


def create_regular_rhythm(name, hypothesis, rr_bounds):
    """
    Creates a new abstraction pattern automata with the properties of a regular
    rhythm, but allows to parameterize the RR limits, the hypothesis observable
    type and the name of the pattern.
    """
    # The hypothesis must be a cardiac rhythm.
    assert issubclass(hypothesis, o.Cardiac_Rhythm)

    def _pair_gconst(pattern, _):
        """
        General constraints to be satisfied when a regular rhythm consists
        of only two beats.
        """
        if pattern.evidence[o.Cardiac_Rhythm]:
            _check_missed_beats(pattern)
            prhythm = pattern.evidence[o.Cardiac_Rhythm][0]
            rhythm = pattern.hypothesis
            # Previous rhythm cannot be a regular rhythm.
            verify(not isinstance(prhythm, o.RegularCardiacRhythm))
            mrr, stdrr = prhythm.meas.rr
            beats = pattern.evidence[o.QRS]
            rr = beats[-1].time.start - beats[0].time.start
            verify(rr in rr_bounds)
            # Avoid duplicate hypotheses with overlapping rhythms.
            if pattern.automata is SINUS_PATTERN:
                verify(C.TACHY_RR.end < rr < C.BRADY_RR.start)
            maxvar = max(C.TMARGIN, min(C.RR_MAX_DIFF, 2.5 * stdrr))
            verify(rr in Iv(mrr - maxvar, mrr + maxvar))
            # Besides being in rhythm, the two beats must share the morphology.
            verify(signal_match(beats[0].shape, beats[1].shape))
            # The amplitude difference is also constrained
            for lead in beats[0].shape:
                if lead in beats[1].shape:
                    samp, qamp = (beats[0].shape[lead].amplitude, beats[1].shape[lead].amplitude)
                    verify(min(samp, qamp) / max(samp, qamp) >= C.MISSED_QRS_MAX_DIFF)
            rhythm.meas = o.CycleMeasurements(
                (rr, stdrr), (prhythm.meas.rt[0], C.QT_ERR_STD), (prhythm.meas.pq[0], C.QT_ERR_STD)
            )

    _qrs_tconst = _get_qrs_tconst(rr_bounds)
    # Automata definition
    automata = PatternAutomata()
    automata.name = name
    automata.Hypothesis = hypothesis
    automata.add_transition(0, 1, o.Cardiac_Rhythm, ENVIRONMENT, _prev_rhythm_tconst, _prev_rhythm_gconst)
    automata.add_transition(1, 2, o.QRS, ENVIRONMENT, _qrs_tconst)
    automata.add_transition(2, 3, o.QRS, ABSTRACTED, _qrs_tconst)
    automata.add_transition(2, 9, o.QRS, ABSTRACTED, _qrs_tconst, _pair_gconst)
    automata.add_transition(3, 4, o.PWave, ABSTRACTED, _p_qrs_tconst)
    automata.add_transition(3, 5, o.TWave, ABSTRACTED, _t_qrs_tconst)
    automata.add_transition(3, 8, o.QRS, ABSTRACTED, _qrs_tconst, _cycle_finished_gconst)
    automata.add_transition(4, 5, o.TWave, ABSTRACTED, _t_qrs_tconst)
    automata.add_transition(5, 6, o.QRS, ABSTRACTED, _qrs_tconst)
    automata.add_transition(5, 8, o.QRS, ABSTRACTED, _qrs_tconst, _cycle_finished_gconst)
    automata.add_transition(6, 7, o.PWave, ABSTRACTED, _p_qrs_tconst)
    automata.add_transition(6, 8, o.TWave, ABSTRACTED, _t_qrs_tconst, _cycle_finished_gconst)
    automata.add_transition(7, 8, o.TWave, ABSTRACTED, _t_qrs_tconst, _cycle_finished_gconst)
    automata.add_transition(8, 6, o.QRS, ABSTRACTED, _qrs_tconst)
    automata.add_transition(8, 8, o.QRS, ABSTRACTED, _qrs_tconst, _cycle_finished_gconst)
    automata.add_transition(9, 10, o.PWave, ABSTRACTED, _p_qrs_tconst)
    automata.add_transition(9, 11, o.TWave, ABSTRACTED, _t_qrs_tconst)
    automata.add_transition(9, 11)
    automata.add_transition(10, 11, o.TWave, ABSTRACTED, _t_qrs_tconst)
    automata.final_states.add(8)
    automata.final_states.add(11)
    automata.abstractions[o.QRS] = (automata.transitions[2], automata.transitions[3])
    automata.obs_proc = _rhythm_obs_proc
    automata.freeze()
    return automata


# Creation of the three regular rhythm patterns.

SINUS_PATTERN = create_regular_rhythm('Sinus Rhythm', o.Sinus_Rhythm, C.SINUS_RR)
BRADYCARDIA_PATTERN = create_regular_rhythm('Bradycardia', o.Bradycardia, C.BRADY_RR)
TACHYCARDIA_PATTERN = create_regular_rhythm('Tachycardia', o.Tachycardia, C.TACHY_RR)


if __name__ == "__main__":
    pass
