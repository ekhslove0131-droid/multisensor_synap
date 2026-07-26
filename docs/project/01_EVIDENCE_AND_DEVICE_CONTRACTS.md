# 근거와 장비 데이터 계약

- 기준일: 2026-07-25
- 목적: 제조사 장비 제약과 연구 근거를 합성 규칙으로 변환할 때의 경계를 정의한다.
- 주의: 실제 SDK 패키지에서 확인한 capability가 이 문서보다 우선한다.

---

## 1. 근거 등급

| 등급 | 의미 | 합성기에 적용하는 방식 |
|---|---|---|
| A | 제조사 공식 SDK·장비 명세 | 샘플링률, 채널, 단위, 측정 모드의 기본 계약 |
| B | 메타분석·체계적 문헌고찰 | population prior의 넓은 방향성과 불확실성 |
| C | 개별 전향·관찰 연구 | 사건 윈도, 멀티모달 결합, 개인화 시나리오 후보 |
| D | 프로젝트 가정 | 명시적으로 `simulation_choice`로 표시하고 설정 가능하게 구현 |

논문 결과가 서로 다르면 평균 방향 하나로 고정하지 않는다. 개인별 반응 계수의 분포를 넓히고, 일부 참여자에게는 반대 방향 또는 무반응을 허용한다.

---

# 2. 장비별 계약

## 2.1 Galaxy Watch8 — Samsung Health Sensor SDK

공식 문서상 연속 tracker와 on-demand tracker를 분리해야 한다.

### 연속 사용 계약

| 신호 | 공식 형태 | 공식 빈도 | 프로젝트 역할 |
|---|---|---:|---|
| 손목 가속도 `watch_acc_*` | raw XYZ | 25 Hz | 손목 활동, 반복 움직임, PPG/EDA 아티팩트 설명 |
| EDA `watch_eda` | raw | 1 Hz | 교감 각성의 관측 축 |
| 심박·IBI `watch_hr`, `watch_ibi` | processed | 1 Hz | Polar H10의 보조·불일치 피처 |
| PPG green/IR/red | raw | 25 Hz | 손목 맥파와 품질·동작 아티팩트 |
| 피부·주변 온도 | processed | event-driven | 느린 변화와 접촉·환경 문맥 |

- EDA 연속 측정은 공식 문서상 **Galaxy Watch8 시리즈 이상**에서 제공된다.
- 피부 온도는 심부 체온이 아니다.
- 피부 온도 native event rate는 실제 SDK 콜백을 측정해 확정한다. 합성 MVP에서는 1 Hz 관측 그리드로 설정하되 `simulation_choice`로 기록한다.

### on-demand 계약

| 신호 | 공식 빈도 | 핵심 연속 데이터에 포함 여부 |
|---|---:|---|
| ECG | 500 Hz | 제외. 선택적 calibration clip만 |
| PPG green/IR/red | 100 Hz | 제외. 선택적 calibration clip만 |
| SpO2 | 단발 processed | 제외 |
| BIA/MF-BIA | 단발 processed | 제외 |

공식 지침은 on-demand 측정을 전경에서, 한 번에 하나만, 보통 30초 동안 사용하도록 하며 연속 tracker와 동시에 측정하면 연속값이 무효가 될 수 있다고 경고한다. 따라서 합성 핵심 타임라인에서는 Watch ECG·100 Hz PPG를 상시 신호처럼 생성하지 않는다.

### 합성기 capability 필드

```yaml
galaxy_watch8:
  sdk_profile: samsung_health_sensor_sdk
  capability_check_required: true
  continuous:
    accelerometer: true
    eda: true
    heart_rate_ibi: true
    ppg_25hz: true
    skin_temperature: true
  on_demand:
    ecg_500hz: optional_calibration_only
    ppg_100hz: optional_calibration_only
```

**공식 출처**

- [Samsung Health Sensor SDK — Data Specifications](https://developer.samsung.com/health/sensor/guide/data-specifications.html)
- [Samsung Health Sensor SDK — API Overview](https://developer.samsung.com/health/sensor/api-reference/overview-summary.html)

---

## 2.2 Polar H10 — Polar BLE SDK

Polar H10은 이 프로젝트의 지속 심장 신호 기준 장비다.

| 신호 | 공식 빈도·범위 | 프로젝트 역할 |
|---|---|---|
| 심박 + RR interval | 1 Hz | 장비 제공 보조 심박·RR |
| ECG | 130 Hz, µV | R-peak, RR/NN, HRV의 1차 원천 |
| 가속도 | 25/50/100/200 Hz, ±2/4/8G | 흉부 움직임, 운동·호흡성 움직임 문맥 |

### 기본 합성 설정

```yaml
polar_h10:
  ecg_rate_hz: 130
  acc_rate_hz: 100
  acc_range_g: 8
  hr_rr_rate_hz: 1
```

- ECG에서 자체 검출한 R-peak/RR과 장비 제공 RR을 별도 열로 유지한다.
- `rr_device_ms`와 `rr_from_ecg_ms`의 지연·불일치를 합성한다.
- H10 ECG 130 Hz는 임상 12유도 ECG가 아니며 이 프로젝트도 임상 대체를 가정하지 않는다.

**공식 출처**

- [Polar BLE SDK — Polar H10 product capabilities](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/PolarH10.md)
- [Polar BLE SDK repository](https://github.com/polarofficial/polar-ble-sdk)

---

## 2.3 Muse S

Muse S의 정확한 세대와 SDK 권한은 사용자가 받은 SDK 문서로 최종 확정한다. 아래 기본값은 공식 Muse S Gen 2 하드웨어 명세를 기반으로 한 시작 계약이다.

| 신호 | 기본 명세 | 프로젝트 역할 |
|---|---|---|
| EEG | 4채널 + 2 AUX, 256 Hz, 12-bit | 주 EEG 축 |
| EEG 위치 | TP9, AF7, AF8, TP10; FPz reference | 채널 이름과 위치 계약 |
| 머리 가속도 | XYZ 52 Hz, ±4G | 머리 움직임·EEG 아티팩트 설명 |
| PPG | IR/IR/red 64 Hz | 선택적 맥파·품질 보조 |
| AUX | 하드웨어 존재 | MVP 제외, capability로만 보존 |

### 기본 합성 설정

```yaml
muse_s:
  hardware_profile: muse_s_gen2_default
  capability_check_required: true
  eeg_rate_hz: 256
  eeg_channels: [TP9, AF7, AF8, TP10]
  accelerometer_rate_hz: 52
  ppg_rate_hz: 64
  ppg_enabled: false
  aux_enabled: false
```

### 중요한 제한

- 제품 하드웨어가 제공하는 신호와 사용자가 받은 SDK가 노출하는 신호는 다를 수 있다.
- 실제 SDK가 gyro, PPG, AUX를 제공하지 않으면 합성 기본 계약에서도 비활성화한다.
- 건식 전극, 이마·귀 주변 위치를 반영해 눈 깜빡임, 턱 근전도, 전극 접촉, 머리 움직임 아티팩트를 강하게 모델링한다.

**공식 출처**

- [Muse S Gen 2 specifications](https://eu.choosemuse.com/pages/muse-s-gen2-offer)

---

# 3. 장비 역할 고정

| 생리·행동 축 | 1차 장비 | 보조 장비 | 이유 |
|---|---|---|---|
| EEG | Muse S | 없음 | 고정 4채널 EEG |
| ECG/RR/HRV | Polar H10 | Watch HR/IBI | H10의 연속 ECG와 RR |
| EDA | Galaxy Watch8 | 없음 | Watch8 연속 EDA |
| 손목 움직임 | Galaxy Watch8 | 없음 | 행동·PPG/EDA 아티팩트 |
| 머리 움직임 | Muse S | 없음 | EEG 아티팩트·머리 반복 움직임 |
| 흉부 움직임 | Polar H10 | 없음 | 활동·호흡성 움직임 문맥 |
| 손목 PPG | Galaxy Watch8 | Muse PPG 선택 | H10 대비 맥파 지연·품질 비교 |
| 피부 온도 | Galaxy Watch8 | 없음 | 느린 문맥 피처 |

장비가 중복 측정하는 값은 평균으로 합치지 않는다. 각 관측값과 품질을 보존하고 `cross_device_disagreement` 파생변수를 만든다.

---

# 4. ASD 관련 연구를 사용하는 방법

## 4.1 EEG 사전정보

2023년 resting-state EEG 메타분석은 41개 연구, 자폐 참여자 1,246명과 비자폐 대조 1,455명을 종합했다. 평균적으로 상대 alpha가 낮고 gamma가 높은 결과가 있었지만, 연구 간 이질성과 방법 차이가 컸다.

### 합성 규칙으로의 번역

잘못된 적용:

```text
ASD 참여자 → alpha 항상 감소, gamma 항상 증가
```

권장 적용:

```text
population prior의 평균 방향을 약하게 이동
+ 개인별 넓은 분산
+ 일부 무반응/반대 방향
+ 나이·수면·눈뜸/눈감음·움직임·EMG 영향
+ 동일한 스펙트럼을 갖는 비사건 hard negative
```

EEG 메타분석은 사건 전조를 직접 설명하지 않는다. 따라서 alpha/gamma 차이는 개인 baseline profile 후보이며, 특정 사건 라벨을 만드는 직접 규칙으로 사용하지 않는다.

- 원문: [Resting-state EEG power differences in autism spectrum disorder: a systematic review and meta-analysis](https://doi.org/10.1038/s41398-023-02681-2)

---

## 4.2 HRV 사전정보

2020년 HRV 메타분석은 34개 연구를 정량 분석했고, ASD 집단에서 평균적으로 낮은 baseline parasympathetic HRV와 RSA, 일부 사회적 스트레스 조건에서 낮은 reactivity를 보고했다.

### 합성 규칙으로의 번역

- `baseline_rmssd`, `baseline_rsa_proxy`, `recovery_speed`의 집단 prior를 약하게 조정할 수 있다.
- 모든 가상 참여자에게 낮은 HRV를 강제하지 않는다.
- 호흡을 직접 측정하지 않으므로 RSA는 확정 생리량이 아니라 `rsa_proxy`로 구분한다.
- 짧은 30초 구간의 주파수 영역 HRV를 정상 지표처럼 생성하지 않는다.
- 활동, 자세, 수면, 호흡, 결측 R-peak가 HRV에 미치는 영향을 함께 생성한다.

- 원문: [Heart rate variability in individuals with autism spectrum disorders: A meta-analysis](https://doi.org/10.1016/j.neubiorev.2020.08.007)

---

## 4.3 웨어러블 기반 사건 전조 연구

### Goodwin et al., 2019

- ASD 청소년 20명
- 심혈관·EDA·가속도
- 과거 3분 데이터로 1분 후 aggression onset 예측
- global model AUC 0.71, person-dependent model AUC 0.84

### Imbiriba et al., 2023

- 4개 입원 병원에서 모집
- 분석 포함 70명
- 심혈관·EDA·움직임
- 로지스틱 회귀가 3분 전 평균 AUROC 약 0.80
- 개인·세션 분리, 시간축 feature extraction의 중요성을 보여줌

### 합성 규칙으로의 번역

이 연구들은 다음 설계를 지지하는 근거로만 사용한다.

1. 30초·60초뿐 아니라 **180초 과거 윈도**를 포함한다.
2. 개인 baseline과 person-dependent 변화량을 만든다.
3. EDA·심장·움직임을 동시에 생성한다.
4. 사건 전조가 항상 존재하지 않도록 한다.
5. 입원 환경의 aggression 연구 결과를 일상적 감각 과부하나 멜트다운으로 일반화하지 않는다.
6. 타깃 명칭은 `multimodal_arousal_episode`처럼 중립적으로 둔다.

- 원문: [Goodwin et al., 2019](https://doi.org/10.1002/aur.2151)
- 원문: [Imbiriba et al., 2023](https://doi.org/10.1001/jamanetworkopen.2023.48898)

---

# 5. 연구 근거에서 허용되는 파라미터

| 범주 | 연구가 제공하는 것 | 합성기에서 허용 | 금지 |
|---|---|---|---|
| EEG | 집단 평균 스펙트럼 차이 후보 | baseline prior의 약한 이동 | ASD 진단 라벨 직접 생성 |
| HRV | 집단 평균 HRV·reactivity 차이 후보 | 개인 baseline·회복 분포 조정 | 낮은 RMSSD를 ASD 사건으로 라벨링 |
| EDA·심장·움직임 | 사건 전 몇 분의 멀티모달 변화 가능성 | 30~180초 전조 시나리오 | 특정 방향을 모든 사건에 강제 |
| 개인화 | 개인별 모델이 유리할 수 있음 | person-specific response coefficient | population model 성능 보장 |
| 예측 성능 | 제한된 연구 환경의 결과 | 테스트 설계 참고 | 합성데이터 성능을 임상 성능으로 주장 |

---

# 6. 중립적 사건 체계

MVP에서 사용할 사건 유형은 다음과 같다.

| `event_type` | 의미 | 학습상 역할 |
|---|---|---|
| `multimodal_arousal_episode` | 여러 생리축이 함께 변하는 사건 | 주 target |
| `repetitive_motion_episode` | 주기적 움직임이 두드러지는 사건 | 별도 target 또는 보조 라벨 |
| `ordinary_physical_activity` | 운동으로 HR·EDA·움직임 증가 | 핵심 hard negative |
| `quiet_cognitive_load` | 움직임은 적고 EEG·자율 변화 가능 | hard negative/별도 시나리오 |
| `sleep_transition` | 수면·기상 전환 | 문맥·hard negative |
| `sensor_artifact_episode` | 생리 변화 없이 측정만 왜곡 | 품질 라벨 |
| `recovery_without_peak` | 올라가다 peak 없이 복귀 | 경계 사례 |
| `false_alarm_like_episode` | 일부 축만 target처럼 변화 | 모델 견고성용 hard negative |

`meltdown`, `aggression`, `sensory_overload` 같은 임상·행동 용어는 원시 모델 타깃 이름으로 사용하지 않는다. 나중에 실제 관찰 라벨과 연결할 때 별도의 매핑·검토 계층을 둔다.

---

# 7. 적용 원칙 요약

1. 장비 명세는 **hard contract**다.
2. 논문 결과는 **soft prior**다.
3. 사건 라벨은 논문 수치나 피처 임계값이 아니라 숨겨진 사건 엔진에서 나온다.
4. 개인차, 무반응, 반대 반응, 하드 네거티브를 반드시 포함한다.
5. 실제 SDK capability를 확보하면 `device_contract.yaml`만 바꾸고 핵심 엔진은 유지한다.
6. 합성데이터는 가능성 시연용이며 실제 임상 정확도를 대신하지 않는다.
