"""
전처리 → 진단 → 모델 비교(CV) → 최종모델 학습 → 테스트 예측까지 한 번에 실행하였습니다.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.config import INDEX_COL, OUT_DIR, SEED, TARGET, TEST_FRAC
from src.data import diagnose, feature_matrix, load_all, scaling_check
from src.evaluate import best_threshold, grouped_cv, inner_oof, metrics
from src.models import get_models

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def holdout_split(df):
    """사출 단위·불량 층화로 TEST_FRAC 만큼 테스트셋 분리 (학습에는 절대 사용하지 않음)."""
    cv = StratifiedGroupKFold(n_splits=round(1 / TEST_FRAC), shuffle=True, random_state=SEED)
    strat = df[TARGET].astype(str) + "_" + df["product"]
    dev_idx, test_idx = next(cv.split(df, strat, df["group"]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    # 1. 데이터 적재 + 진단
    df = load_all()
    diag = diagnose(df)
    diag.to_csv(OUT_DIR / "data_diagnosis.csv", index=False, encoding="utf-8-sig")
    scaling_check().to_csv(OUT_DIR / "scaling_check.csv", index=False, encoding="utf-8-sig")
    print("[1] 데이터 진단\n", diag.T, "\n")

    # 2. 개발/테스트 분리
    dev, test = holdout_split(df)
    print(f"[2] dev {len(dev)}행 (불량 {dev[TARGET].sum()}) / test {len(test)}행 (불량 {test[TARGET].sum()})\n")

    # 3. 모델 × 특성셋 비교 (dev 에서 반복 그룹 CV)
    y = dev[TARGET].to_numpy()
    groups = dev["group"].to_numpy()
    products = dev["product"].to_numpy()
    records, oofs = [], {}
    for use_cavity in (False, True):
        X = feature_matrix(dev, use_cavity)
        fs = "process+cavity" if use_cavity else "process_only"
        for name, model in get_models().items():
            recs, oof = grouped_cv(model, X, y, groups, products)
            for r in recs:
                r.update(model=name, features=fs)
            records += recs
            oofs[(name, fs)] = oof
            print(f"  done {name:13s} {fs}  ({time.time() - t0:.0f}s)")

    cv = pd.DataFrame(records)
    cv.to_csv(OUT_DIR / "cv_folds.csv", index=False, encoding="utf-8-sig")
    metric_cols = ["PR_AUC", "ROC_AUC", "F1", "Precision", "Recall", "Recall@5%", "Recall@10%", "Recall@20%"]
    summary = cv.groupby(["scope", "features", "model"])[metric_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(OUT_DIR / "cv_summary.csv", encoding="utf-8-sig")
    print("\n[3] CV 요약 (평균)\n", cv.groupby(["scope", "features", "model"])[metric_cols].mean().round(3), "\n")

    # 4. 최종모델 선정: 전체 scope PR-AUC 평균 최대
    allscope = cv[cv.scope == "all"].groupby(["model", "features"])["PR_AUC"].mean()
    best_model, best_fs = allscope.idxmax()
    print(f"[4] 최종모델: {best_model} / {best_fs} (CV PR-AUC {allscope.max():.3f})")

    use_cavity = best_fs == "process+cavity"
    X_dev, X_test = feature_matrix(dev, use_cavity), feature_matrix(test, use_cavity)
    model = get_models()[best_model]
    thr = best_threshold(y, inner_oof(model, X_dev, y, groups, SEED))
    final = clone(model).fit(X_dev, y)

    # 5. 테스트 예측
    p_test = final.predict_proba(X_test)[:, 1]
    y_test = test[TARGET].to_numpy()
    test_m = {"model": best_model, "features": best_fs, "threshold": thr,
              "all": metrics(y_test, p_test, thr)}
    for prod in ["cn7", "rg3"]:
        m = test["product"].to_numpy() == prod
        test_m[prod] = metrics(y_test[m], p_test[m], thr)
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_m, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = test[[INDEX_COL, "product", "shot_id", "cavity", TARGET]].copy()
    pred["fail_prob"] = p_test
    pred["pred"] = (p_test >= thr).astype(int)
    pred["inspect_rank"] = pred["fail_prob"].rank(ascending=False, method="first").astype(int)
    pred.sort_values("inspect_rank").to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 오류분석용 dev OOF 저장
    oof = dev.copy()
    oof["oof_prob"] = oofs[(best_model, best_fs)]
    oof["oof_pred"] = (oof["oof_prob"] >= thr).astype(int)
    oof["error"] = np.select([(oof[TARGET] == 1) & (oof.oof_pred == 0), (oof[TARGET] == 0) & (oof.oof_pred == 1)],
                             ["FN", "FP"], "OK")
    oof.to_csv(OUT_DIR / "oof_dev.csv", index=False, encoding="utf-8-sig")

    print("\n[5] 테스트 성능\n", pd.DataFrame({k: v for k, v in test_m.items() if isinstance(v, dict)}).round(3))
    print(f"\n완료 ({time.time() - t0:.0f}s) → {OUT_DIR}")


if __name__ == "__main__":
    main()
