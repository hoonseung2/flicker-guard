# 데이터 준비 진행 노트 (DatasetSynth 운영 단계)

> design spec(`docs/superpowers/specs/2026-08-02-dataset-synth-design.md`)이 "파이프라인 완성 후의 별도 운영 작업"으로 미뤄둔 단계의 실제 진행 기록. `data/`는 전체가 `.gitignore` 대상(대용량 바이너리라 커밋 안 함)이라, 여기 있는 경로/스크립트 사용법만 있으면 언제든 재현 가능하다.

## 1. DAVIS 데이터 확보

- **소스**: `https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip` (davischallenge.org 공식 미러)
- **버전**: DAVIS 2017 TrainVal 480p, 794MB, 90개 시퀀스 (train 30 + val 20 기준 공식 문서, 실제 압축 해제 결과 90개 디렉터리 확인됨), 총 6,208프레임 (시퀀스당 평균 69프레임)
- **로컬 경로**: `data/raw/DAVIS/` (압축 해제 완료). `JPEGImages/480p/<sequence-name>/*.jpg` 형태.
- **라이선스**: 확인 안 됨 — README §9의 Flickerformer 라이선스 리스크와 같은 급으로, 실제 제품에 쓰기 전 확인 필요.

## 2. JPEG 시퀀스 → mp4 변환

- **스크립트**: `scripts/davis_to_clips.py` (`training/` 패키지 밖의 1회성 유틸리티, SDD 리뷰 대상 아님)
- **fps 가정**: 24fps로 인코딩 (DAVIS 자체엔 타이밍 메타데이터 없음, 관례적 값 — 파이프라인은 인코딩한 fps를 그대로 읽어서 쓰므로 정확성에 영향 없음)
- **사용법**: `python scripts/davis_to_clips.py --davis-root data/raw/DAVIS --output <출력폴더> --fps 24 [--sequences 이름1 이름2 ...]` (`--sequences` 생략 시 90개 전체 변환)
- 지금까지 변환한 것: `bear, camel, dance-twirl, boat, car-turn` → `data/clips/`

## 3. Clean 소스 사전 스크리닝

- **스크립트**: `scripts/screen_clean_clips.py` — 변환된 mp4를 원본 그대로(합성 전) Detector에 돌려서, 이미 위험으로 걸리는 클립을 "clean 소스로 부적합"으로 걸러냄
- **사용법**: `python scripts/screen_clean_clips.py --clips-dir <mp4폴더> --profiles-dir <프로파일폴더>`
- **6개 프로파일 전체로 스크리닝한 결과** (5개 테스트): `car-turn`(kr에서만 걸림), `dance-twirl`(6개 전부 걸림) → clean 3/5
- **itu만으로 스크리닝한 결과**: `dance-twirl`만 걸림 → clean 4/5 (car-turn은 itu 25% 기준으론 통과)
- → **스크리닝 기준은 실제로 사용할 프로파일 세트와 일치시켜야 함** (전체 6개로 걸러낼 필요 없음, 우리가 학습에 안 쓸 프로파일 기준으로 탈락시키는 건 불필요하게 손실)

## 4. 프로파일 축소 결정: itu 하나만 사용

**결정**: DatasetSynth 학습 데이터 생성은 `itu` 프로파일 하나만 사용. `data/profiles_active/itu.json` (itu.json 사본, `configs/profiles/`는 Detector 런타임용으로 6개 그대로 유지 — 절대 안 건드림).

**이유**:
1. 6개 프로파일 중 실제로 다른 파라미터는 `max_area_ratio`뿐 (`kr=0.10`, 나머지 5개=`0.25`). 나머지 4개 필드(`max_flashes_per_second`, `general_flash_dark_threshold`, `general_flash_delta_threshold`, `red_saturation_ratio_threshold`)는 6개 전부 동일값. 즉 지금 스키마(주파수+면적 두 숫자만 인코딩, README §9의 알려진 한계)로는 실질적으로 **2개 그룹**(kr vs 나머지 5개)뿐.
2. `kr`은 한국 웹사이트에만 적용되는 좁은 범위라 학습 데이터에서 제외하기로 함.
3. 남은 5개(itu/jp/netflix/ofcom/w3c) 중 itu 선택 — README §3에서 전체 탐지 파이프라인 구조 자체가 ITU-R BT.1702-04를 참조 구조로 삼고 있어서 가장 근거 있는 대표값.
4. **주의**: 이건 "지금 스키마가 표현 못 하는 지역별 세부 규칙(Ofcom 어두운 장면 규칙, 일본 패턴 밀도 규칙)이 아직 없어서" 생기는 임시 통폐합임. 스키마가 확장되면 이 결정을 다시 풀어야 할 수 있음.
5. Detector 런타임(`configs/profiles/`) 자체는 배포 지역에 따라 6개 다 필요 — 이 결정은 DatasetSynth 학습 데이터 생성 범위에만 해당, Detector 자체 변경 아님.

## 5. 실측치 (itu 프로파일, 3클립: bear/boat/camel 기준)

- **합성+저장**: 3클립 × 2패턴(general/red) × 1샘플 = 6조합 → **77.18초**, 재시도 0회 (전부 첫 시도 성공)
- **디스크**: 650MB (샘플당 clean 52% + degraded 48%; clean은 같은 클립의 모든 조합에서 100% 동일 파일 — I-5로 알려진 중복, 코드는 안 고치고 트레이드오프로 인정하기로 함)
- **프레임당 비용**: 합성+저장 약 0.156초/프레임(패턴당), 스크리닝(탐지만) 약 0.072초/프레임
- **90개 전체 확장 시 추정**: 스크리닝 ~7.5분 + 합성 ~26분(통과율 80% 가정 시) ≈ **총 30~35분**, 용량 **약 10~16GB** (통과율에 따라 변동)

## 6. DAVIS 90개 전체 실행 결과 (완료)

- **변환**: 90개 전체 → `data/clips_all/` (228MB)
- **스크리닝** (itu 기준): **61 clean / 29 flagged** (67.8%) — flagged 목록은 `scripts/screen_clean_clips.py --clips-dir data/clips_all --profiles-dir data/profiles_active`로 재현 가능
- **배치 합성** (itu, general+red, 61클립 × 2패턴 = 122조합): **116 성공 / 6 실패**, 재시도 소진은 0건
  - 실패 6건 전부 "clip_too_short" — `dog-agility`(25프레임), `judo`(34프레임), `rollerblade`(35프레임)이 itu 최소 요구치(37프레임 @ 24fps)에 못 미침. 크래시 없이 정확한 사유로 기록됨.
- **최종 산출물**: `data/synthetic/` — 116개 샘플 폴더, **8.9GB**
- 사전 추정(10~16GB)보다 실제로는 낮게 나옴 — 통과율(67.8%)이 초기 5클립 표본(80%)보다 낮았지만, 짧은 클립 6개가 조기 실패해서 상쇄됨.

## 7. 다음 단계

- [x] DAVIS 90개 전체: 스크리닝(itu 기준) → 변환 → 배치 실행
- [ ] Mitigator 학습 설계 — 단, README 아키텍처상 `Detector → PriorCalc → Mitigator`인데 **PriorCalc가 아직 없음**. Mitigator 설계 전에 PriorCalc부터 다룰지 결정 필요.
