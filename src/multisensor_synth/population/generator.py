from __future__ import annotations

import numpy as np

from multisensor_synth.config.models import ProjectConfig
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.domain.types import ParticipantRecord
from multisensor_synth.population.baselines import (
    sample_beta,
    sample_lognormal,
    sample_truncated_normal,
)


def generate_participants(
    config: ProjectConfig, seeds: SeedTree, run_id: str
) -> list[ParticipantRecord]:
    population_rng = seeds.rng("population")
    population_shift = float(population_rng.normal(0, 0.15))
    population = config.population
    archetype_probabilities = population.response_archetypes.model_dump()
    archetype_names = tuple(archetype_probabilities)
    archetype_weights = tuple(archetype_probabilities.values())
    rows: list[ParticipantRecord] = []

    for index in range(1, config.run.participant_count + 1):
        person_id = f"{population.participant_id_prefix}{index:03d}"
        namespace = f"person/{person_id}/baseline"
        seed = seeds.child_seed(namespace)
        rng = np.random.default_rng(seed)
        age = sample_truncated_normal(population.age_years, rng)
        sex = str(
            rng.choice(
                population.biological_sex.categories,
                p=population.biological_sex.probabilities,
            )
        )
        archetype = str(rng.choice(archetype_names, p=archetype_weights))

        cardiac = population.baseline_profiles.cardiac
        resting_hr = sample_truncated_normal(cardiac.resting_hr_bpm, rng)
        resting_z = (resting_hr - cardiac.resting_hr_bpm.mean) / cardiac.resting_hr_bpm.std
        rmssd = sample_lognormal(cardiac.rmssd_ms, rng) * np.exp(-0.08 * resting_z)
        eeg = population.baseline_profiles.eeg

        rows.append(
            ParticipantRecord(
                run_id=run_id,
                person_id=person_id,
                age_years=float(np.float32(age)),
                biological_sex=sex,
                response_archetype=archetype,
                resting_hr_bpm_truth=float(np.float32(resting_hr)),
                rmssd_ms_truth=float(np.float32(rmssd)),
                eeg_aperiodic_exponent_truth=float(
                    np.float32(sample_truncated_normal(eeg.aperiodic_exponent, rng))
                ),
                eeg_relative_alpha_truth=float(
                    np.float32(sample_beta(eeg.relative_alpha, rng))
                ),
                eeg_relative_gamma_truth=float(
                    np.float32(sample_beta(eeg.relative_gamma, rng))
                ),
                eda_tonic_level_us_truth=float(
                    np.float32(
                        sample_lognormal(
                            population.baseline_profiles.eda.tonic_level_us,
                            rng,
                        )
                    )
                ),
                recovery_capacity_truth=float(
                    np.float32(
                        np.clip(
                            sample_beta(
                                population.baseline_profiles.recovery_capacity,
                                rng,
                            )
                            + 0.03 * population_shift,
                            0,
                            1,
                        )
                    )
                ),
                opposite_response_tendency_truth=float(
                    np.float32(
                        np.clip(
                            population.allow_opposite_direction_probability
                            + rng.normal(0, 0.04),
                            0,
                            1,
                        )
                    )
                ),
                no_response_tendency_truth=float(
                    np.float32(
                        np.clip(
                            population.allow_no_response_probability + rng.normal(0, 0.04),
                            0,
                            1,
                        )
                    )
                ),
                seed=seed,
            )
        )
    return rows
