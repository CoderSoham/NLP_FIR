"""Where the two analyses disagree.

A pure comparison of two dicts, so it lives outside the module that loads
eight models -- and so the cases it has already got wrong can be tested in
milliseconds rather than never.

A disagreement is information, not an error. Two independent methods reaching
different conclusions about an emergency call is exactly the case a human
should look at, so it is reported rather than resolved by picking a winner.
"""


# The LLM has a 'critical' band the classical scorer does not. Folding it in
# is only sound applied to *both* sides: when the classical value came from the
# model itself, folding one side turned 'critical' vs 'critical' into a
# reported disagreement.
SEVERITY_SCALE = {'critical': 'high'}


def compare_classifications(classical, llm_record, second_opinion=True):
    """Where the classical pipeline and the LLM disagree.

    A disagreement is information, not an error. Two independent methods
    reaching different conclusions about an emergency call is exactly the case
    a human should look at, so it is reported rather than resolved by picking a
    winner.

    `second_opinion` says whether the BART classifiers actually ran. When they
    did not, type and severity on both sides came from the same model, and
    comparing a value with itself is not a comparison. Weapons and location
    still are: those come from the keyword matcher and spaCy, which run either
    way.
    """
    if not llm_record:
        return []

    found = []
    if second_opinion:
        llm_type = llm_record.get('incident_type')
        if llm_type and llm_type not in ('unknown', 'other') and llm_type != classical.get('emergency_type'):
            found.append({
                'field': 'emergency_type',
                'classical': classical.get('emergency_type'),
                'llm': llm_type,
            })

        llm_sev = llm_record.get('severity')
        normalised = SEVERITY_SCALE.get(llm_sev, llm_sev)
        classical_sev = classical.get('severity')
        if (normalised and normalised != 'unknown'
                and normalised != SEVERITY_SCALE.get(classical_sev, classical_sev)):
            found.append({
                'field': 'severity',
                'classical': classical_sev,
                'llm': llm_sev,
            })

    # The keyword matcher flags weapons on any mention; the LLM is asked to
    # ignore figures of speech. This is the disagreement that matters most.
    keyword_weapons = bool((classical.get('response') or {}).get('signals', {}).get('weapons'))
    llm_weapons = bool(llm_record.get('weapons'))  # now [{item, quote}, ...]
    if keyword_weapons != llm_weapons:
        found.append({
            'field': 'weapons',
            'classical': 'mentioned' if keyword_weapons else 'not mentioned',
            'llm': ([w.get('item') for w in llm_record.get('weapons') or []]
                    or 'none present'),
        })

    classical_location = classical.get('probable_location')
    llm_location = llm_record.get('location')
    if bool(classical_location) != bool(llm_location):
        found.append({
            'field': 'location',
            'classical': classical_location,
            'llm': llm_location,
        })

    return found
