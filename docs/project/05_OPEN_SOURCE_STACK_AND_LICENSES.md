# 오픈소스 스택과 라이선스 정책

- 기준일: 2026-07-25
- 목적: 합성데이터 생성기의 핵심 경로에 사용할 라이브러리와 역할을 고정한다.
- 법률 자문이 아니며, 배포 전 실제 LICENSE와 dependency tree를 다시 검토해야 한다.

---

# 1. 선택 원칙

1. 오픈소스가 사건 의미와 라벨을 자동으로 정하도록 하지 않는다.
2. 오픈소스는 신호 생성·처리·검증의 검증된 부품으로 사용한다.
3. 핵심 runtime은 MIT, BSD, Apache 계열을 우선한다.
4. BSL, GPL, AGPL 또는 상업 조건이 있는 패키지는 승인 없이 핵심 dependency로 넣지 않는다.
5. 실제 데이터가 없는 초기 단계에서는 TimeGAN 같은 학습형 생성기를 핵심으로 사용하지 않는다.
6. 버전은 lock 파일로 고정하며, 문서의 버전 숫자는 시작점일 뿐이다.

---

# 2. 핵심 스택

## 2.1 Python·수치·데이터

| 프로젝트 | 역할 | 라이선스 | 사용 위치 |
|---|---|---|---|
| Python 3.12 | 실행 환경 | PSF | 전체 |
| NumPy | RNG, 배열, 확률과정 | BSD-3-Clause | 전체 |
| SciPy | signal, stats, interpolation | BSD-3-Clause | 신호·분포 |
| pandas | 소규모 표·메타데이터 | BSD-3-Clause | 설정·리포트 |
| PyArrow | 대용량 Parquet·dataset partition | Apache-2.0 | 저장·읽기 |
| Pydantic | 엄격한 설정 모델 | MIT | config |
| Pandera | DataFrame schema 검증 | MIT | validation |
| scikit-learn | sanity model, 거리·분할 | BSD-3-Clause | 검증 |
| statsmodels | 선택적 시계열·통계 | BSD-3-Clause | latent/statistics |

대용량 canonical table의 핵심 저장은 pandas 단일 DataFrame보다 PyArrow dataset writer를 우선한다.

---

## 2.2 EEG

### MNE-Python

- 역할: EEG 채널 정보, filtering, PSD, epoch/clip, FIF export
- 라이선스: BSD-3-Clause
- 기준일 최신 릴리스로 확인된 버전: 1.12.1
- 핵심 사용: 예

[MNE-Python repository](https://github.com/mne-tools/mne-python)

### NeuroDSP

- 역할: periodic/aperiodic neural time-series 구성, oscillation·burst·1/f simulation, DSP
- 라이선스: Apache-2.0
- 확인 버전: 2.3.0
- 핵심 사용: 예

[NeuroDSP documentation](https://neurodsp-tools.github.io/)

### 사용 분담

```text
NeuroDSP: 이상적 EEG 성분 생성
→ 프로젝트 코드: 채널별 결합·사건 효과·개인차
→ 프로젝트 artifact model: EOG/EMG/contact/motion
→ MNE: 처리·feature·FIF export
```

MNE의 분석 함수가 생성 정답을 정하지 않는다.

---

## 2.3 ECG·PPG·EDA

### NeuroKit2

- 역할:
  - `ecg_simulate`
  - `ppg_simulate`
  - `eda_simulate`
  - ECG/PPG/EDA processing과 feature validation
- 라이선스: MIT
- 확인 버전: 0.2.13
- 핵심 사용: 예

[NeuroKit2 repository](https://github.com/neuropsychology/NeuroKit)

### 사용 분담

```text
latent beat timing / event effect
→ NeuroKit2 waveform primitive
→ 프로젝트 morphology·device model
→ artifact·clock·packet model
→ NeuroKit2 또는 자체 extractor로 재계산 검증
```

`NeuroKit2` 기본값을 그대로 사용하지 않고 H10 130 Hz, Watch PPG 25 Hz, Watch EDA 1 Hz 계약에 맞춘다.

---

## 2.4 증강·아티팩트

### tsaug

- 역할: crop, time warp, drift, dropout 등 일반 시계열 augmentation 참고·선택 적용
- 라이선스: Apache-2.0
- 상태: 선택 dependency

[tsaug repository](https://github.com/arundo/tsaug)

아티팩트 truth가 필요한 핵심 경로는 자체 명시적 artifact generator를 사용한다. 무작위 augmentation만 적용하면 어떤 교란이 들어갔는지 정답을 잃을 수 있다.

---

## 2.5 장비 추상화·재생 테스트

### BrainFlow

- 역할: Synthetic Board, Streaming Board, biosensor API 추상화, 향후 replay integration test
- 라이선스: MIT
- 상태: 선택 dependency

[BrainFlow repository](https://github.com/brainflow-dev/brainflow)

사용자가 이미 제조사 SDK를 확보했으므로 BrainFlow를 실제 장비 연결의 필수 계층으로 만들지 않는다. 합성 스트림을 실제 수집기 형태로 재생하는 테스트 도구로만 고려한다.

---

# 3. 선택적 2단계 생성 모델

## 3.1 SynthCity

- 라이선스: Apache-2.0
- 시계열 모델: TimeGAN, FourierFlows, TimeVAE
- 평가 지표 포함
- 사용 시점: 실제 시계열 seed data가 충분히 모인 이후
- 핵심 runtime 의존성: 아니오

[SynthCity repository](https://github.com/vanderschaarlab/synthcity)

### 허용 용도

- 실제 데이터의 분포를 학습한 생성 모델과 규칙 기반 생성기 비교
- TSTR/TRTS와 통계 거리 평가
- 특정 feature table의 보조 증강 실험

### 금지 용도

- 실제 데이터 없이 ASD 패턴을 자동 생성한다고 가정
- 사건 정답과 단계 라벨을 TimeGAN에 맡김
- 모델 내부에서 생성된 패턴을 임상 근거로 해석

## 3.2 SDV

- 기능: tabular·sequential data synthesis와 평가
- 현재 공개 배포 라이선스: Business Source License
- 핵심 runtime 의존성: **금지**
- 사용 조건: 라이선스 검토와 명시 승인 후 별도 실험 환경

[SDV repository](https://github.com/sdv-dev/SDV)

SDV가 기술적으로 유용하더라도 초기 프로젝트의 상업적 확장 가능성을 고려해 permissive core에서 제외한다.

---

# 4. 사용하지 않을 도구

| 도구·접근 | 이유 |
|---|---|
| GPT/LLM 기반 시계열 생성 | 신호 물리·라벨 정합성 통제 어려움, 사용자 요구에서 제외 |
| EEGPT | 특징 모델이지 이 프로젝트의 규칙 기반 합성기 요구와 다름 |
| 범용 tabular GAN을 초기 핵심으로 사용 | 실제 seed data 없음, 사건 인과·라벨 통제 불가 |
| 폐기·비활성 프로젝트를 핵심 dependency로 사용 | 유지보수 위험 |
| 라이선스 불명확한 Muse reverse-engineering 코드 | SDK 확보 상태에서 불필요한 법적·기술적 위험 |
| GPL/AGPL package를 승인 없이 runtime에 포함 | 배포 정책 영향 가능 |

---

# 5. SDK와 오픈소스의 관계

이 합성기 runtime은 Samsung, Muse, Polar SDK binary를 요구하지 않는다.

SDK가 제공하는 것은 다음 계약이다.

- 실제 채널 이름
- 데이터 타입
- 샘플링률
- 단위
- timestamp 의미
- packet 구조
- capability 차이

합성기는 이 계약을 `device_contract.yaml`로 복제한다. 향후 실기기 adapter가 같은 canonical schema를 출력하도록 한다.

제조사 SDK 자체의 라이선스와 재배포 조건은 별도다. 저장소에 SDK 파일을 커밋하지 않는다.

---

# 6. 권장 dependency 그룹

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy",
  "scipy",
  "pyarrow",
  "pandas",
  "pydantic>=2",
  "pandera[pandas]",
  "mne",
  "neurodsp",
  "neurokit2",
  "scikit-learn",
  "jinja2",
]

[project.optional-dependencies]
augmentation = ["tsaug"]
replay = ["brainflow"]
research-generators = ["synthcity"]
dev = [
  "pytest",
  "pytest-cov",
  "hypothesis",
  "ruff",
  "mypy",
]
```

정확한 버전은 구현 시작 시 호환성 검증 후 `uv.lock`에 고정한다.

---

# 7. 라이선스 승인 정책

## 자동 허용 후보

- MIT
- BSD-2-Clause
- BSD-3-Clause
- Apache-2.0
- ISC
- PSF

## 수동 승인 필요

- BSL/BUSL
- GPL/LGPL/AGPL
- SSPL
- source-available
- custom academic/non-commercial
- 라이선스 불명확

## CI 산출물

- `uv.lock`
- `licenses.csv`
- CycloneDX 또는 SPDX SBOM
- dependency name/version/license/source
- 승인 예외 목록

추천 도구는 구현 시 선택하되, 결과 파일 형식과 검증 테스트를 요구한다.

---

# 8. 버전 고정 전략

1. Python minor version 고정
2. 핵심 패키지는 `uv.lock` 사용
3. 생성 결과 manifest에 package version 기록
4. golden test는 dependency 변경 시 차이를 검토
5. major/minor 업데이트는 데이터 분포 변화 검사 후 승인
6. 연구용 optional group은 core CI에서 제외 가능

---

# 9. 도구별 책임 매트릭스

| 책임 | 프로젝트 코드 | MNE | NeuroDSP | NeuroKit2 | tsaug | BrainFlow | SynthCity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 개인 baseline | ● |  |  |  |  |  |  |
| latent state | ● |  |  |  |  |  |  |
| event/label | ● |  |  |  |  |  |  |
| EEG primitive | ● | ○ | ● |  |  |  |  |
| ECG/PPG/EDA primitive | ● |  |  | ● |  |  |  |
| device artifact truth | ● |  |  | ○ | ○ |  |  |
| EEG processing/export | ○ | ● | ○ |  |  |  |  |
| cardiac/EDA validation | ○ |  |  | ● |  |  |  |
| schema validation | ● |  |  |  |  |  |  |
| replay adapter | ○ |  |  |  |  | ● |  |
| learned generator comparison |  |  |  |  |  |  | ● |

`●` 주 책임, `○` 보조.

---

# 10. 구현 승인 기준

- core install에 SDV/SynthCity/BrainFlow/tsaug가 없어도 `quick`과 `mvp` 생성이 가능하다.
- 핵심 dependency의 라이선스가 permissive 정책을 만족한다.
- 제조사 SDK 파일을 저장소에 포함하지 않는다.
- 라이선스와 버전이 manifest에 기록된다.
- optional dependency가 없을 때 명확한 메시지를 낸다.
- 오픈소스 함수의 default sampling rate를 장비 계약 대신 사용하지 않는다.
