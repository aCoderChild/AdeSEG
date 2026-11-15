"""Validate the three-fold registry and patient-balanced training schedule."""
from scripts.crossfit_training_contract_3fold_v1 import fold_split, patient_balanced_schedule
from scripts.preflight_warm_crossfit_training_v1 import SEED, EPOCHS, UPDATES_PER_PATIENT
from scripts.preflight_bounded_crossfit_training_v1 import require


def fold_definition(registry, fold, rows):
    require(type(fold) is int and 1 <= fold <= 3, 'Invalid fold')
    train, validation, test = fold_split(fold)
    expected = {'fold': fold, 'train': sorted(train), 'validation': sorted(validation),
                'test': sorted(test), 'training_order': patient_balanced_schedule(
                    list(rows.values()), fold, SEED, EPOCHS, UPDATES_PER_PATIENT)}
    require(len(registry['folds']) == 3 and registry['folds'][fold - 1] == expected,
            'Fold identities or training schedule differ')
    return train, validation, test, expected['training_order']
