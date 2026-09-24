# 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

## 실행
```bash
pip install -r requirements.txt
python run_all.py
```
전처리 → 데이터 진단 → 모델 비교(반복 그룹 CV) → 최종모델 학습 → 테스트 예측까지 자동 실행된다 (CPU 약 10분).

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `data/` | 제공 데이터 원본 (labeled/unlabeled × CN7/RG3) |
| `src/config.py` | 경로·시드·CV 설정 |
| `src/data.py` | 적재, 사출(shot)·캐비티 구조 복원, 데이터 진단 |
| `src/models.py` | 비교 모델 (로지스틱 베이스라인, RandomForest, LightGBM) |
| `src/evaluate.py` | 그룹 CV, 학습 fold 내부 임계값 선정, 검사 우선순위 지표 |
| `run_all.py` | 전체 파이프라인 |

## 산출물 (`outputs/`)
| 파일 | 내용 |
|---|---|
| `data_diagnosis.csv` | 제품별 불량률, 사출 구조, 라벨 충돌 등 진단표 |
| `scaling_check.csv` | 라벨/비라벨 데이터가 각각 따로 표준화되었음을 보이는 표 |
| `cv_folds.csv`, `cv_summary.csv` | 모델 × 특성셋 × 제품별 CV 성능 |
| `test_metrics.json` | 홀드아웃 테스트 성능 |
| `test_predictions.csv` | **테스트데이터 예측결과** (불량확률, 판정, 검사 우선순위) |
| `oof_dev.csv` | 개발셋 OOF 예측 + FN/FP 표시 (오류분석용) |

## 핵심 설계
- **2캐비티 구조**: 공정값이 동일한 연속 2행 = 한 번의 사출에서 나온 제품 2개. 같은 사출이 학습/평가에 나뉘지 않도록 사출 단위 `StratifiedGroupKFold` 사용.
- **테스트셋**: 사출 단위·불량 층화로 20% 분리, 모델 선택·임계값 선정에 사용하지 않음.
- **임계값**: 각 학습 fold 내부 3-fold 예측으로만 F1 최대 임계값 선정 (평가 fold 누수 방지).
- **지표**: 극단적 불균형이므로 PR-AUC 를 주지표로, F1 과 Recall@상위k%(검사 우선순위)를 함께 보고.
- 비라벨 데이터는 라벨 데이터와 별도로 표준화되어 있어 동일 공간에서 결합하지 않음.
