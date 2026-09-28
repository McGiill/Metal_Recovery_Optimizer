"""Module 3 — 최적 공정 레시피 추천.

소유자: P4  (이 파일은 P4만 수정한다)
계약: docs/INTERFACE.md

스펙 #5 요구사항: 목표 함수 정의 / 제약 조건 설정 / 최적화 수행
스펙 #7 권장:   Grid·Random Search 외 베이지안 최적화, 유전 알고리즘
"""

# 실측 범위 기반 제약 조건 (센티널 제외 후)
BOUNDS = {
    "acid_l": (402.0, 1499.0),
    "temp_c": (40.1, 94.9),
    "time_hr": (1.0, 8.0),
}

TARGET = {"ni_recovery_pct": 95.0, "co_recovery_pct": 90.0}

# ⚠ 목표 동시 달성 배치가 420건 중 6건뿐이다 (Ni≥95 단독 22건 / Co≥90 단독 9건).
#   Grid Search 결과를 그대로 내놓으면 학습 데이터가 없는 영역을 외삽하게 된다.
#   feasible=False 케이스 처리가 필수이며, 동료평가 질문
#   "도출된 레시피가 현장에서 실제 구현 가능한 값인지 어떻게 보장하는가" 가 이 지점을 찌른다.


def recommend(model, composition: dict, target: dict, bounds: dict) -> dict:
    """주어진 원료 성분에서 목표 회수율을 달성하는 공정 조건을 탐색한다.

    Returns:
        {"acid_l":…, "temp_c":…, "time_hr":…,
         "pred_ni":…, "pred_co":…, "feasible": bool}
    """
    raise NotImplementedError
