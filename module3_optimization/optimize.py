"""Module 3 — 최적 공정 레시피 추천 (공개 진입점).

소유자: P4  (이 파일은 P4만 수정한다)
계약: docs/INTERFACE.md, module3_optimization/README.md 4장

스펙 #5 요구사항: 목표 함수 정의 / 제약 조건 설정 / 최적화 수행
스펙 #7 권장:   Grid·Random Search 외 베이지안 최적화, 유전 알고리즘

목표치 · 범위는 config.yaml 로 옮겼다. 실제 구현은 recommend.py.

참고 (사전 측정, 센티널 제외 후 실측 범위):
  acid_l 402–1499 / temp_c 40.1–94.9 / time_hr 1–8
  박스 제약은 데이터 사전 범위(400–1500 / 40–95 / 1–8)를 쓰고,
  실측 범위 바깥은 적용 범위 필터(domain.py)가 담당한다.

⚠ 목표 동시 달성 배치가 420건 중 6건뿐이다 (Ni≥95 단독 22건 / Co≥90 단독 9건).
  Grid Search 결과를 그대로 내놓으면 학습 데이터가 없는 영역을 외삽하게 된다.
  status = infeasible / out_of_domain 처리가 필수이며, 동료평가 질문
  "도출된 레시피가 현장에서 실제 구현 가능한 값인지 어떻게 보장하는가" 가 이 지점을 찌른다.
"""

from module3_optimization.recommend import recommend

__all__ = ["recommend"]
