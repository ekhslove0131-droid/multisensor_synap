from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RangeFloat(StrictModel):
    min: float
    max: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("range min must be <= max")
        return self


class RangeInt(StrictModel):
    min: int
    max: int

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("range min must be <= max")
        return self


class TruncatedNormal(StrictModel):
    distribution: Literal["truncated_normal"]
    mean: float
    std: float = Field(gt=0)
    min: float
    max: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("distribution min must be <= max")
        return self


class LogNormal(StrictModel):
    distribution: Literal["lognormal"]
    median: float = Field(gt=0)
    geometric_std: float = Field(gt=1)


class BetaDistribution(StrictModel):
    distribution: Literal["beta"]
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class DiscreteUniform(StrictModel):
    distribution: Literal["discrete_uniform"]
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("distribution min must be <= max")
        return self


class RunSettings(StrictModel):
    seed: int = Field(ge=0)
    start_time_utc: datetime
    duration_sec: int = Field(gt=0)
    duration_days: int | None = Field(default=None, gt=0)
    participant_count: int = Field(gt=0)
    canonical_rate_hz: int = Field(default=1)
    timezone_context: str = "UTC"
    chunk_duration_sec: int = Field(default=3600, gt=0)
    deterministic: bool = True
    fail_on_warning: bool = False

    @model_validator(mode="after")
    def validate_run_contract(self) -> Self:
        if self.canonical_rate_hz != 1:
            raise ValueError("canonical_rate_hz must equal 1")
        if self.start_time_utc.utcoffset() != UTC.utcoffset(self.start_time_utc):
            raise ValueError("run.start_time_utc must be timezone-aware UTC")
        if self.duration_days is not None and self.duration_sec != self.duration_days * 86_400:
            raise ValueError("duration_sec must match duration_days")
        try:
            ZoneInfo(self.timezone_context)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone_context must name an available timezone") from exc
        return self


class StorageSettings(StrictModel):
    output_root: Path = Path("output")
    parquet_compression: Literal["zstd", "snappy", "none"] = "zstd"
    partition_by: tuple[Literal["person_id", "date_utc"], ...] = (
        "person_id",
        "date_utc",
    )
    write_truth: bool = True
    write_native_clips: bool = True
    write_full_native_streams: bool = False
    write_mne_fif_for_eeg: bool = True
    estimated_size_guard_gb: float = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_goal_one_storage(self) -> Self:
        if self.write_full_native_streams:
            raise ValueError("write_full_native_streams must be false by default")
        return self


class EegStream(StrictModel):
    enabled: bool = True
    sample_rate_hz: int
    channels: tuple[str, ...]
    unit: str
    adc_bits: int = Field(gt=0)
    reference: str
    mains_frequency_hz: Literal[50, 60]


class AccelerometerStream(StrictModel):
    enabled: bool = True
    sample_rate_hz: int
    range_g: int | None = None
    allowed_sample_rates_hz: tuple[int, ...] | None = None
    allowed_ranges_g: tuple[int, ...] | None = None
    axes: tuple[str, ...]
    mode: str | None = None


class PpgStream(StrictModel):
    enabled: bool
    sample_rate_hz: int
    channels: tuple[str, ...]
    mode: str | None = None


class AuxStream(StrictModel):
    enabled: bool = False


class MuseSConfig(StrictModel):
    enabled: bool = True
    hardware_profile: str
    capability_check_required: bool = True
    eeg: EegStream
    accelerometer: AccelerometerStream
    ppg: PpgStream
    aux: AuxStream

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if (
            self.eeg.sample_rate_hz != 256
            or self.eeg.channels != ("TP9", "AF7", "AF8", "TP10")
            or self.eeg.unit != "uV"
        ):
            raise ValueError("Muse S EEG contract requires 256 Hz, fixed channels, and uV")
        if (
            self.accelerometer.sample_rate_hz != 52
            or self.accelerometer.range_g != 4
            or self.accelerometer.axes != ("x", "y", "z")
        ):
            raise ValueError("Muse S accelerometer contract requires 52 Hz XYZ at +/-4 g")
        if self.ppg.sample_rate_hz != 64:
            raise ValueError("Muse S PPG contract requires 64 Hz")
        return self


class EcgStream(StrictModel):
    enabled: bool = True
    sample_rate_hz: int
    unit: str
    leads: tuple[str, ...]


class HrRrStream(StrictModel):
    enabled: bool = True
    sample_rate_hz: int
    rr_unit: str


class PolarH10Config(StrictModel):
    enabled: bool = True
    ecg: EcgStream
    hr_rr: HrRrStream
    accelerometer: AccelerometerStream

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.ecg.sample_rate_hz != 130 or self.ecg.unit != "uV":
            raise ValueError("Polar H10 ECG contract requires 130 Hz and uV")
        if self.hr_rr.sample_rate_hz != 1 or self.hr_rr.rr_unit != "ms":
            raise ValueError("Polar H10 HR/RR contract requires 1 Hz and RR ms")
        if self.accelerometer.sample_rate_hz not in (25, 50, 100, 200):
            raise ValueError("Polar H10 ACC rate must be one of 25, 50, 100, 200 Hz")
        if self.accelerometer.range_g not in (2, 4, 8):
            raise ValueError("Polar H10 ACC range must be one of 2, 4, 8 g")
        if self.accelerometer.allowed_sample_rates_hz != (25, 50, 100, 200):
            raise ValueError("Polar H10 allowed ACC rates must remain 25, 50, 100, 200 Hz")
        if self.accelerometer.allowed_ranges_g != (2, 4, 8):
            raise ValueError("Polar H10 allowed ACC ranges must remain 2, 4, 8 g")
        return self


class ContinuousStream(StrictModel):
    enabled: bool = True
    mode: str
    sample_rate_hz: int
    unit: str | None = None
    axes: tuple[str, ...] | None = None
    channels: tuple[str, ...] | None = None


class TemperatureStream(StrictModel):
    enabled: bool = True
    mode: Literal["continuous_event"]
    synthetic_observation_rate_hz: int
    unit: str = Field(validation_alias=AliasChoices("unit", "units"))


class OnDemandStream(StrictModel):
    enabled: bool = False
    policy: str


class OnDemandConfig(StrictModel):
    ecg_500hz: OnDemandStream
    ppg_100hz: OnDemandStream

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        for stream in (self.ecg_500hz, self.ppg_100hz):
            if stream.policy != "calibration_only":
                raise ValueError("Watch on-demand policy must be calibration_only")
        if self.ecg_500hz.enabled and self.ppg_100hz.enabled:
            raise ValueError("Watch on-demand ECG and PPG cannot both be enabled")
        return self


class GalaxyWatch8Config(StrictModel):
    enabled: bool = True
    sdk_profile: str
    capability_check_required: bool = True
    accelerometer: ContinuousStream
    eda: ContinuousStream
    heart_rate_ibi: ContinuousStream
    ppg: ContinuousStream
    skin_temperature: TemperatureStream
    on_demand: OnDemandConfig

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.accelerometer.sample_rate_hz != 25 or self.accelerometer.axes != (
            "x",
            "y",
            "z",
        ):
            raise ValueError("Galaxy Watch8 ACC contract requires 25 Hz XYZ")
        if self.eda.sample_rate_hz != 1 or self.eda.unit != "uS":
            raise ValueError("Galaxy Watch8 EDA contract requires 1 Hz and uS")
        if self.heart_rate_ibi.sample_rate_hz != 1:
            raise ValueError("Galaxy Watch8 HR/IBI contract requires 1 Hz")
        if self.ppg.sample_rate_hz != 25 or self.ppg.channels != ("green", "ir", "red"):
            raise ValueError("Galaxy Watch8 PPG contract requires 25 Hz green/ir/red")
        if self.skin_temperature.synthetic_observation_rate_hz != 1:
            raise ValueError("Galaxy Watch8 temperature simulation choice requires 1 Hz")
        return self


class DevicesConfig(StrictModel):
    muse_s: MuseSConfig
    polar_h10: PolarH10Config
    galaxy_watch8: GalaxyWatch8Config


class CategoryProbabilities(StrictModel):
    categories: tuple[Literal["female", "male"], ...]
    probabilities: tuple[float, ...]

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        if len(self.categories) != len(self.probabilities):
            raise ValueError("biological sex category and probability lengths must match")
        if any(value < 0 or value > 1 for value in self.probabilities):
            raise ValueError("biological sex probabilities must be in [0,1]")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError("biological sex probabilities must sum to 1")
        return self


class CardiacBaseline(StrictModel):
    resting_hr_bpm: TruncatedNormal
    rmssd_ms: LogNormal


class EegBaseline(StrictModel):
    aperiodic_exponent: TruncatedNormal
    relative_alpha: BetaDistribution
    relative_gamma: BetaDistribution


class EdaBaseline(StrictModel):
    tonic_level_us: LogNormal


class BaselineProfiles(StrictModel):
    cardiac: CardiacBaseline
    eeg: EegBaseline
    eda: EdaBaseline
    recovery_capacity: BetaDistribution


class ArchetypeProbabilities(StrictModel):
    eeg_first: float = Field(ge=0, le=1)
    autonomic_first: float = Field(ge=0, le=1)
    motor_first: float = Field(ge=0, le=1)
    quiet_internal: float = Field(ge=0, le=1)
    motor_dominant: float = Field(ge=0, le=1)
    partial_response: float = Field(ge=0, le=1)
    non_responder: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> Self:
        total = (
            self.eeg_first
            + self.autonomic_first
            + self.motor_first
            + self.quiet_internal
            + self.motor_dominant
            + self.partial_response
            + self.non_responder
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("response archetype probabilities must sum to 1")
        return self


class PopulationConfig(StrictModel):
    participant_id_prefix: str = "P"
    age_years: TruncatedNormal
    biological_sex: CategoryProbabilities
    baseline_profiles: BaselineProfiles
    response_archetypes: ArchetypeProbabilities
    research_prior_strength: float = Field(ge=0, le=1)
    allow_opposite_direction_probability: float = Field(ge=0, le=1)
    allow_no_response_probability: float = Field(ge=0, le=1)


class CircadianConfig(StrictModel):
    enabled: bool = True
    amplitude_variation: float = Field(ge=0, le=1)


class NormalHours(StrictModel):
    mean: float
    std: float = Field(gt=0)


class SleepConfig(StrictModel):
    enabled: bool = True
    bedtime_local_hour: NormalHours
    duration_hours: NormalHours


class ActivityConfig(StrictModel):
    daily_bouts: RangeInt


class MealsConfig(StrictModel):
    enabled: bool = True
    daily_count: int = Field(ge=0)


class ContextConfig(StrictModel):
    circadian: CircadianConfig
    sleep: SleepConfig
    ordinary_activity: ActivityConfig
    meals: MealsConfig


class ClipWindow(StrictModel):
    pre_sec: int = Field(ge=0)
    post_sec: int = Field(ge=0)


class PhaseDurations(StrictModel):
    pre_early_sec: RangeInt
    pre_late_sec: RangeInt
    onset_sec: RangeInt
    peak_sec: RangeInt
    recovery_early_sec: RangeInt
    recovery_late_sec: RangeInt
    post_sec: RangeInt

    @model_validator(mode="after")
    def validate_required_phases(self) -> Self:
        ranges = (
            self.pre_early_sec,
            self.pre_late_sec,
            self.onset_sec,
            self.peak_sec,
            self.recovery_early_sec,
            self.recovery_late_sec,
            self.post_sec,
        )
        if any(bounds.min < 0 or bounds.max < 0 for bounds in ranges):
            raise ValueError("phase durations must be non-negative")
        if self.onset_sec.max <= 0 or self.peak_sec.max <= 0:
            raise ValueError("target onset and peak phases must be positive")
        return self


class HardNegativeProbabilities(StrictModel):
    ordinary_physical_activity: float = Field(ge=0, le=1)
    quiet_cognitive_load: float = Field(ge=0, le=1)
    sensor_artifact_episode: float = Field(ge=0, le=1)
    recovery_without_peak: float = Field(ge=0, le=1)
    false_alarm_like_episode: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> Self:
        total = (
            self.ordinary_physical_activity
            + self.quiet_cognitive_load
            + self.sensor_artifact_episode
            + self.recovery_without_peak
            + self.false_alarm_like_episode
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("hard-negative type probabilities must sum to 1")
        return self


class HardNegativesConfig(StrictModel):
    ratio_to_target: float = Field(ge=0)
    types: HardNegativeProbabilities


class EventsConfig(StrictModel):
    target_type: Literal["multimodal_arousal_episode"]
    target_events_per_day: DiscreteUniform
    minimum_gap_between_target_sec: int = Field(ge=0)
    clip_window: ClipWindow
    phases: PhaseDurations
    intensity: BetaDistribution
    modality_delays_sec: dict[Literal["eeg", "cardiac", "eda", "motor"], RangeInt]
    hard_negatives: HardNegativesConfig
    matched_baseline_clips_per_day: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_modality_delays(self) -> Self:
        expected = {"eeg", "cardiac", "eda", "motor"}
        if set(self.modality_delays_sec) != expected:
            raise ValueError("modality delay keys must be eeg, cardiac, eda, motor")
        return self


class GlobalArtifacts(StrictModel):
    enabled: bool = True
    random_missing_probability_per_hour: float = Field(ge=0, le=1)
    block_dropout_events_per_day: RangeInt


class ClockArtifacts(StrictModel):
    initial_offset_ms: dict[Literal["muse_s", "polar_h10", "galaxy_watch8"], RangeInt]
    drift_ppm: dict[Literal["muse_s", "polar_h10", "galaxy_watch8"], RangeInt]
    jitter_ms_std: dict[Literal["muse_s", "polar_h10", "galaxy_watch8"], float]

    @model_validator(mode="after")
    def validate_device_keys(self) -> Self:
        expected = {"muse_s", "polar_h10", "galaxy_watch8"}
        if set(self.initial_offset_ms) != expected:
            raise ValueError("clock offset device keys are incomplete")
        if set(self.drift_ppm) != expected:
            raise ValueError("clock drift device keys are incomplete")
        if set(self.jitter_ms_std) != expected:
            raise ValueError("clock jitter device keys are incomplete")
        if any(value < 0 for value in self.jitter_ms_std.values()):
            raise ValueError("clock jitter standard deviation must be non-negative")
        return self


class EegArtifacts(StrictModel):
    blink_events_per_min: RangeInt
    jaw_emg_probability_per_min: float = Field(ge=0, le=1)
    electrode_pop_probability_per_hour: float = Field(ge=0, le=1)
    channel_dropout_probability_per_hour: float = Field(ge=0, le=1)
    line_noise_hz: Literal[50, 60]


class EcgArtifacts(StrictModel):
    baseline_wander_strength: RangeFloat
    motion_artifact_probability_per_min: float = Field(ge=0, le=1)
    missed_peak_probability: float = Field(ge=0, le=1)
    extra_peak_probability: float = Field(ge=0, le=1)


class PpgArtifacts(StrictModel):
    motion_coupling_strength: RangeFloat
    contact_loss_probability_per_hour: float = Field(ge=0, le=1)


class EdaArtifacts(StrictModel):
    motion_spike_probability_per_min: float = Field(ge=0, le=1)
    flatline_probability_per_hour: float = Field(ge=0, le=1)


class AccelerometerArtifacts(StrictModel):
    bias_std_g: float = Field(ge=0)
    scale_error_std: float = Field(ge=0)
    clipping_probability_per_hour: float = Field(ge=0, le=1)


class ArtifactsConfig(StrictModel):
    global_: GlobalArtifacts = Field(alias="global")
    clocks: ClockArtifacts
    eeg: EegArtifacts
    ecg: EcgArtifacts
    ppg: PpgArtifacts
    eda: EdaArtifacts
    accelerometer: AccelerometerArtifacts


class PersonalBaselineConfig(StrictModel):
    enabled: bool = True
    warmup_sec: int = Field(gt=0)
    method: Literal["rolling_robust"]
    lookback_sec: int = Field(gt=0)
    exclude_contexts: tuple[str, ...]
    exclude_low_quality: bool = True


class EegFeatures(StrictModel):
    frequency_bands_hz: dict[
        Literal["delta", "theta", "alpha", "beta", "gamma"], tuple[float, float]
    ]
    connectivity_enabled: bool
    aperiodic_enabled: bool

    @model_validator(mode="after")
    def validate_frequency_bands(self) -> Self:
        expected = {"delta", "theta", "alpha", "beta", "gamma"}
        if set(self.frequency_bands_hz) != expected:
            raise ValueError("EEG frequency band keys are incomplete")
        for name, bounds in self.frequency_bands_hz.items():
            if len(bounds) != 2 or bounds[0] < 0 or bounds[0] >= bounds[1]:
                raise ValueError(f"invalid EEG frequency band bounds: {name}")
        return self


class HrvFeatures(StrictModel):
    frequency_domain_min_window_sec: int = Field(gt=0)
    entropy_min_window_sec: int = Field(gt=0)


class FeaturesConfig(StrictModel):
    windows_sec: tuple[int, ...]
    causal_only: bool
    minimum_coverage_default: float = Field(ge=0, le=1)
    personal_baseline: PersonalBaselineConfig
    eeg: EegFeatures
    hrv: HrvFeatures
    registry_path: Path

    @model_validator(mode="after")
    def validate_causal_windows(self) -> Self:
        if not self.causal_only:
            raise ValueError("features.causal_only must be true")
        if any(window <= 0 for window in self.windows_sec):
            raise ValueError("feature windows must be positive")
        return self


class PhaseLabels(StrictModel):
    baseline: Literal[0]
    pre_early: Literal[1]
    pre_late: Literal[2]
    onset: Literal[3]
    peak: Literal[4]
    recovery_early: Literal[5]
    recovery_late: Literal[6]
    post: Literal[7]


class LabelsConfig(StrictModel):
    phases: PhaseLabels
    forecast_horizons_sec: tuple[int, ...]
    include_intensity_target: bool
    include_artifact_labels: bool
    label_source: Literal["synthetic_rule_v1"]
    overlap_policy: Literal["multi_axis"]


class SplitsConfig(StrictModel):
    strategy: Literal["by_person"]
    train_fraction: float = Field(ge=0, le=1)
    validation_fraction: float = Field(ge=0, le=1)
    test_fraction: float = Field(ge=0, le=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> Self:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        return self


class ValidationConfig(StrictModel):
    schema_: bool = Field(alias="schema")
    invariants: bool
    feature_recomputation: bool
    leakage_checks: bool
    statistical_checks: bool
    simple_model_sanity_check: bool
    generate_html_report: bool


class ProjectConfig(StrictModel):
    schema_version: Literal["1.0"]
    project: Literal["multisensor_synth"]
    profile: Literal["quick", "mvp", "golden"]
    run: RunSettings
    storage: StorageSettings
    devices: DevicesConfig
    population: PopulationConfig
    context: ContextConfig
    events: EventsConfig
    artifacts: ArtifactsConfig
    features: FeaturesConfig
    labels: LabelsConfig
    splits: SplitsConfig
    validation: ValidationConfig
