from copy import deepcopy

from scripts.assess_region_conditioned_candidate import development_decision


def make_audit():
    def metric(dice, pixels=0):
        return {'dice': dice, 'predicted_pixels': pixels}
    base = [
        {'sequence':'v1', 'frame':1, 'split':'validation', 'present':True,
         'baseline':metric(.5), 'refined':metric(.8)},
        {'sequence':'v1', 'frame':2, 'split':'validation', 'present':False,
         'baseline':metric(0, 10), 'refined':metric(0, 9)},
    ]
    arms = {name:{'rows':deepcopy(base)} for name in ('region','global','shuffled','mask_only')}
    for name, score in [('global',.7),('shuffled',.6),('mask_only',.55)]:
        arms[name]['rows'][0]['refined']['dice'] = score
    return {'arms': arms}


def test_region_must_beat_all_matched_controls():
    audit = make_audit()
    assert development_decision(audit)['eligible']
    audit['arms']['global']['rows'][0]['refined']['dice'] = .85
    result = development_decision(audit)
    assert not result['eligible'] and not result['checks']['beats_global_present_dice']


def test_region_cannot_improve_dice_by_expanding_empty_false_positive():
    audit = make_audit()
    audit['arms']['region']['rows'][1]['refined']['predicted_pixels'] = 11
    result = development_decision(audit)
    assert not result['eligible'] and not result['checks']['no_more_empty_predicted_pixels_per_video']
