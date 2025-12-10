# -*- coding: utf-8 -*-
"""
PID별 텍스트 요약 + 시나리오 서술 + 임베딩 벡터 생성 스크립트

사용 예시:
    python generate_pid_summary_and_embeddings.py \
        --data_path data/pid_panel.csv \
        --meta_path data/meta_information.csv \
        --theme_path data/theme_tags.csv \
        --scenario_path data/scenario_rules.json \
        --output_text_path outputs/pid_texts.csv \
        --output_embed_path outputs/pid_embeddings.npy
"""

import argparse
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# 1. 메타 & 테마 로딩
# ---------------------------------------------------------------------

def load_meta(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, encoding="cp949")
    meta = meta.rename(columns={c: c.strip() for c in meta.columns})
    meta = meta.set_index("컬럼(물리)명 ")

    def parse_value_map(raw):
        if not isinstance(raw, str):
            return None
        parts = [p.strip() for p in raw.split(',')]
        m = {}
        for p in parts:
            if ':' in p:
                k, v = p.split(':', 1)
                m[k.strip()] = v.strip()
        return m or None

    meta["value_map"] = meta["데이터 값"].apply(parse_value_map)
    return meta


def load_theme_tags(theme_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(theme_path)
    tag_dict = {}
    for _, row in df.iterrows():
        col = row["column"]
        themes = str(row["theme"]).split(";")
        themes = [t.strip() for t in themes if t and t.strip()]
        tag_dict[col] = themes
    return tag_dict


def load_scenario_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# 2. 유틸 함수 (설명 축약, 시계열 요약 등)
# ---------------------------------------------------------------------

def short_desc(text: Any, max_len: int = 70) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    t = text.splitlines()[0]
    t = t.split("①")[0] if "①" in t else t
    t = t.replace('"', "").strip()
    if len(t) > max_len:
        t = t[:max_len] + "..."
    return t


def summarize_flag_over_time(pid_df: pd.DataFrame, col: str, meta: pd.DataFrame) -> str:
    if col not in meta.index or col not in pid_df.columns:
        return ""
    info = meta.loc[col]
    logical = info["컬럼(논리)명"]
    desc = short_desc(info["컬럼설명"])

    s = pid_df.set_index("chasu")[col].fillna(0)
    if s.max() == 0:
        return ""  # 전 기간 해당 없음이면 생략

    ones = s[s == 1]
    first_t = ones.index.min()
    last_t = ones.index.max()
    cnt = ones.shape[0]

    if cnt == 1:
        return f"{logical}({desc})에 한 차례 해당되며, {first_t} 차수에서 관측되었습니다."
    else:
        return (f"{logical}({desc})에 총 {cnt}개 차수에서 해당되며, "
                f"처음은 {first_t}, 마지막은 {last_t} 차수입니다.")


def summarize_numeric_trend(pid_df: pd.DataFrame, col: str, meta: pd.DataFrame) -> str:
    if col not in meta.index or col not in pid_df.columns:
        return ""
    info = meta.loc[col]
    logical = info["컬럼(논리)명"]

    s = pid_df.set_index("chasu")[col].dropna()
    if s.empty:
        return ""
    s = s.sort_index()
    first_t, last_t = s.index[0], s.index[-1]
    first_val, last_val = s.iloc[0], s.iloc[-1]
    min_val, max_val = s.min(), s.max()

    return (f"{logical}는 {first_t} 차수에 {int(first_val)}에서 시작해 "
            f"{last_t} 차수에 {int(last_val)}로 변화하였으며, "
            f"전체 기간 최소 {int(min_val)}, 최대 {int(max_val)} 수준으로 기록되었습니다.")


# ---------------------------------------------------------------------
# 3. 블록(변수군) 기반 요약
# ---------------------------------------------------------------------

def build_block_summary(pid_df: pd.DataFrame,
                        meta: pd.DataFrame,
                        theme_tags: Dict[str, List[str]]) -> List[str]:
    pid_df = pid_df.sort_values("chasu")
    parts: List[str] = []

    # 기본 인적/가구 정보
    pid = pid_df["PID_encrypt"].iloc[0]
    chasu_min = pid_df["chasu"].min()
    chasu_max = pid_df["chasu"].max()
    age_min = pid_df["AGE"].min()
    age_max = pid_df["AGE"].max()
    sex = pid_df["SEX"].mode().iloc[0]
    fnum = pid_df["FNUM"].mode().iloc[0]
    sido = pid_df["SIDO"].mode().iloc[0]
    sigungu = pid_df["SIGUNGU"].mode().iloc[0]

    parts.append(
        f"이 아동(PID={pid})은 {chasu_min} 차수에 처음 관측된 이후 {chasu_max} 차수까지 데이터가 존재합니다. "
        f"관측 기간 동안 연령은 {age_min}세에서 {age_max}세로 증가하였으며, "
        f"{sido} {sigungu}에 거주하는 {int(fnum)}인 가구의 구성원입니다."
    )

    # 사각지대 주요 변수들
    key_V = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V9", "V10", "V11",
             "V12", "V13", "V14", "V15", "V16", "V18", "V19",
             "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
             "V29", "V30", "V31"]
    v_sents = [summarize_flag_over_time(pid_df, c, meta) for c in key_V]
    v_sents = [s for s in v_sents if s]
    if v_sents:
        parts.append("사각지대 관련 주요 변수는 다음과 같습니다. " + " ".join(v_sents))

    # 사회보장/소득
    key_S_flag = ["S13", "S14", "S22", "S23", "S24", "S27", "S28", "S29", "S30",
                  "S35", "S36", "S37", "S38", "S39", "S40"]
    key_S_num = ["S33", "S34"]
    s_sents = [summarize_flag_over_time(pid_df, c, meta) for c in key_S_flag]
    s_sents += [summarize_numeric_trend(pid_df, c, meta) for c in key_S_num]
    s_sents = [s for s in s_sents if s]
    if s_sents:
        parts.append("사회보장 및 소득 관련 정보는 다음과 같습니다. " + " ".join(s_sents))

    # 위기아동 지표
    key_N = ["N01", "N02", "N03", "N04", "N05", "N06", "N08", "N09", "N10",
             "SV01", "SV02", "SV03"]
    n_sents = [summarize_flag_over_time(pid_df, c, meta) for c in key_N]
    n_sents = [s for s in n_sents if s]
    if n_sents:
        parts.append("위기아동 관련 핵심 지표는 다음과 같습니다. " + " ".join(n_sents))

    # TYPE
    if "TYPE" in pid_df.columns:
        s_type = pid_df.set_index("chasu")["TYPE"]
        changes = s_type[s_type != s_type.shift(1)]
        if not changes.empty:
            desc = "; ".join([f"{idx} 차수: 유형 {val}" for idx, val in changes.items()])
            parts.append(f"분석모형 TYPE 값은 시점별로 {desc}로 변경되었습니다.")

    return parts


# ---------------------------------------------------------------------
# 4. 이벤트 추출 & 시나리오 감지
# ---------------------------------------------------------------------

def extract_events(pid_df: pd.DataFrame,
                   meta: pd.DataFrame,
                   theme_tags: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    각 PID에 대해 (변수, chasu) 수준의 이벤트 리스트 생성
    이벤트 타입:
      - flag_on: 0 -> 1 전환
      - numeric_drop / numeric_rise: 큰 변화
    """
    events: List[Dict[str, Any]] = []
    df = pid_df.sort_values("chasu").set_index("chasu")

    for col in df.columns:
        if col in ["PID_encrypt"]:
            continue
        if col not in meta.index:
            continue

        dtype = meta.loc[col]["데이터타입"]
        themes = theme_tags.get(col, [])

        s = df[col]

        if dtype == "범주형":
            # 0 -> 1 전환만 이벤트로 본다
            prev = s.shift(1).fillna(0)
            onset = (prev == 0) & (s == 1)
            for t in s.index[onset]:
                events.append({
                    "time": t,
                    "column": col,
                    "type": "flag_on",
                    "themes": themes
                })

        elif dtype == "수치형":
            # 큰 폭 감소/증가 이벤트 (여기서는 간단하게 30% 기준 예시)
            if s.dropna().empty:
                continue
            prev = s.shift(1)
            ratio = (s - prev) / prev
            drop = (ratio <= -0.3)  # 30% 이상 감소
            rise = (ratio >= 0.3)   # 30% 이상 증가
            for t in s.index[drop]:
                events.append({
                    "time": t,
                    "column": col,
                    "type": "numeric_drop",
                    "themes": themes,
                    "ratio": float(ratio.loc[t])
                })
            for t in s.index[rise]:
                events.append({
                    "time": t,
                    "column": col,
                    "type": "numeric_rise",
                    "themes": themes,
                    "ratio": float(ratio.loc[t])
                })

    return events


def detect_scenarios(events: List[Dict[str, Any]],
                     pid_df: pd.DataFrame,
                     scenario_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    매우 단순한 룰 기반 시나리오 감지 (프로토타입)
    """
    scenarios: List[Dict[str, Any]] = []

    # chasu를 숫자 index로 취급 (예: 201801 -> 0, 201804 -> 1 ...)
    chasu_order = sorted(pid_df["chasu"].unique())
    chasu_to_idx = {c: i for i, c in enumerate(chasu_order)}

    # 도우미: 특정 변수 집합에 속한 이벤트만 뽑기
    def filter_events_by_vars(vars_list):
        return [e for e in events if e["column"] in vars_list]

    # health_to_employment
    if "health_to_employment" in scenario_rules:
        rule = scenario_rules["health_to_employment"]
        health_events = filter_events_by_vars(rule["health_vars"])
        emp_events = filter_events_by_vars(rule["employment_vars"])
        w = rule["window_chasu"]

        if health_events and emp_events:
            # 가장 이른 health 이벤트 이후 w 이내 emp 이벤트가 있으면 1개 시나리오로 본다
            t_h = min(chasu_to_idx[e["time"]] for e in health_events)
            t_e = min(chasu_to_idx[e["time"]] for e in emp_events)
            if t_e >= t_h and t_e <= t_h + w:
                first_health_time = chasu_order[t_h]
                window_desc = f"약 {w}개 차수(약 {w*3}개월)"
                text = rule["template"].format(
                    health_vars=", ".join(rule["health_vars"]),
                    employment_vars=", ".join(rule["employment_vars"]),
                    t_health_first=first_health_time,
                    window_desc=window_desc
                )
                scenarios.append({
                    "id": "health_to_employment",
                    "name": rule["name"],
                    "text": text
                })

    # income_drop_to_benefit
    if "income_drop_to_benefit" in scenario_rules and "S34" in pid_df.columns:
        rule = scenario_rules["income_drop_to_benefit"]
        df = pid_df.sort_values("chasu").set_index("chasu")
        s_income = df["S34"].astype(float)
        prev = s_income.shift(1)
        ratio = (prev - s_income) / prev  # 감소 비율
        drop = (ratio >= rule["drop_ratio"])
        if drop.any():
            t_drop = drop.idxmax()
            # 같은 window 내 benefit_vars flag_on 이벤트 수를 센다
            income_idx = chasu_to_idx[t_drop]
            w = rule["window_chasu"]
            benefit_events = [e for e in events if e["column"] in rule["benefit_vars"]
                              and e["type"] == "flag_on"
                              and chasu_to_idx[e["time"]] >= income_idx
                              and chasu_to_idx[e["time"]] <= income_idx + w]
            num_new_benefits = len({e["column"] for e in benefit_events})
            if num_new_benefits >= rule["min_new_benefits"]:
                window_desc = f"{w}개 차수 이내"
                text = rule["template"].format(
                    t_income_drop=t_drop,
                    drop_percent=int(ratio.loc[t_drop] * 100),
                    benefit_vars=", ".join(rule["benefit_vars"]),
                    num_new_benefits=num_new_benefits,
                    window_desc=window_desc
                )
                scenarios.append({
                    "id": "income_drop_to_benefit",
                    "name": rule["name"],
                    "text": text
                })

    # debt_cluster
    if "debt_cluster" in scenario_rules:
        rule = scenario_rules["debt_cluster"]
        df = pid_df.sort_values("chasu").set_index("chasu")
        w = rule["window_chasu"]
        debt_cols = [c for c in rule["debt_vars"] if c in df.columns]
        if debt_cols:
            # 각 chasu별 활성 debt 변수 개수
            active = (df[debt_cols] == 1).sum(axis=1)
            # rolling window 내 최대값이 min_active 이상인지 확인
            arr = active.values
            for i in range(len(arr)):
                window_max = arr[i:i+w].max()
                if window_max >= rule["min_active"]:
                    t_start = df.index[i]
                    window_desc = f"{w}개 차수 내"
                    text = rule["template"].format(
                        debt_vars=", ".join(debt_cols),
                        min_active=rule["min_active"],
                        window_desc=window_desc
                    )
                    scenarios.append({
                        "id": "debt_cluster",
                        "name": rule["name"],
                        "text": text
                    })
                    break

    # household_structure_plus_childrisk
    if "household_structure_plus_childrisk" in scenario_rules:
        rule = scenario_rules["household_structure_plus_childrisk"]
        h_events = [e for e in events if e["column"] in rule["household_vars"]]
        c_events = [e for e in events if e["column"] in rule["childrisk_vars"]]
        if h_events and c_events:
            w = rule["window_chasu"]
            window_desc = f"{w}개 차수 이내"
            # 가장 가까운 쌍이 window 안에 있으면 시나리오 존재로 본다
            for he in h_events:
                ih = chasu_to_idx[he["time"]]
                if any(abs(chasu_to_idx[ce["time"]] - ih) <= w for ce in c_events):
                    text = rule["template"].format(
                        household_vars=", ".join(rule["household_vars"]),
                        childrisk_vars=", ".join(rule["childrisk_vars"]),
                        window_desc=window_desc
                    )
                    scenarios.append({
                        "id": "household_structure_plus_childrisk",
                        "name": rule["name"],
                        "text": text
                    })
                    break

    return scenarios


# ---------------------------------------------------------------------
# 5. PID 전체 요약 텍스트 생성
# ---------------------------------------------------------------------

def describe_pid(pid_df: pd.DataFrame,
                 meta: pd.DataFrame,
                 theme_tags: Dict[str, List[str]],
                 scenario_rules: Dict[str, Any]) -> str:
    block_parts = build_block_summary(pid_df, meta, theme_tags)
    events = extract_events(pid_df, meta, theme_tags)
    scenarios = detect_scenarios(events, pid_df, scenario_rules)

    scenario_texts = [s["text"] for s in scenarios]
    if scenario_texts:
        block_parts.append("전 기간을 종합한 시나리오 관점에서 보면, " +
                           " ".join(scenario_texts))

    return " ".join(block_parts)


# ---------------------------------------------------------------------
# 6. 전체 PID에 대해 텍스트 + 임베딩 생성
# ---------------------------------------------------------------------

def build_all_pid_texts(df: pd.DataFrame,
                        meta: pd.DataFrame,
                        theme_tags: Dict[str, List[str]],
                        scenario_rules: Dict[str, Any]) -> pd.DataFrame:
    texts = []
    for pid, g in df.groupby("PID_encrypt"):
        g = g.drop_duplicates(subset=["PID_encrypt", "chasu"])
        text = describe_pid(g, meta, theme_tags, scenario_rules)
        texts.append({"PID_encrypt": pid, "text": text})
    return pd.DataFrame(texts)


def embed_texts(texts: List[str],
                model_name: str = "intfloat/multilingual-e5-base") -> np.ndarray:
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    return vecs


# ---------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="PID × chasu panel 데이터 csv 경로")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="meta_information.csv 경로")
    parser.add_argument("--theme_path", type=str, required=True,
                        help="theme_tags.csv 경로")
    parser.add_argument("--scenario_path", type=str, required=True,
                        help="scenario_rules.json 경로")
    parser.add_argument("--output_text_path", type=str, required=True,
                        help="PID별 텍스트 요약 csv 출력 경로")
    parser.add_argument("--output_embed_path", type=str, required=True,
                        help="PID별 임베딩 numpy 출력 경로(.npy)")
    parser.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-base")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, encoding="cp949")
    meta = load_meta(args.meta_path)
    theme_tags = load_theme_tags(args.theme_path)
    scenario_rules = load_scenario_rules(args.scenario_path)

    pid_text_df = build_all_pid_texts(df, meta, theme_tags, scenario_rules)
    pid_text_df.to_csv(args.output_text_path, index=False, encoding="utf-8-sig")

    vecs = embed_texts(pid_text_df["text"].tolist(), model_name=args.embed_model)
    np.save(args.output_embed_path, vecs)

    print(f"총 {len(pid_text_df)}개 PID에 대해 텍스트 및 임베딩을 생성했습니다.")
    print(f"텍스트: {args.output_text_path}")
    print(f"임베딩: {args.output_embed_path}")


if __name__ == "__main__":
    main()
