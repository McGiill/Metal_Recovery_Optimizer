"""index.html + scene.js + data.json 을 단일 파일 standalone.html 로 묶는다.

    python docs/modeling/build_standalone.py

Streamlit 의 components.html() 은 내용을 iframe 의 srcdoc 으로 넣는다.
srcdoc iframe 은 origin 이 null 이라 상대 경로 요청이 전부 막힌다 —
<script src="./scene.js"> 도, fetch("./data.json") 도 실패한다.
그래서 둘 다 문서 안에 넣어 둔 사본이 필요하다.
three.js 는 절대 URL(cdn.jsdelivr.net)이라 그대로 둔다.

data.json 을 다시 뽑았으면(export_data.py) 이 스크립트도 다시 돌린다.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_TAG = '<script type="module" src="./scene.js"></script>'
FETCH = 'fetch("./data.json").then(r => r.json())'


def main() -> int:
    page = (HERE / "index.html").read_text(encoding="utf-8")
    scene = (HERE / "scene.js").read_text(encoding="utf-8")
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))

    if SRC_TAG not in page:
        raise SystemExit(f"index.html 에서 스크립트 태그를 못 찾았다: {SRC_TAG}")
    if FETCH not in scene:
        raise SystemExit(f"scene.js 에서 데이터 로드 지점을 못 찾았다: {FETCH}")

    # 상대 경로 fetch 를 인라인 상수로 바꾼다
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    scene = scene.replace(FETCH, "Promise.resolve(__SCENE_DATA__)")
    inline = (
        '<script type="module">\n'
        f"const __SCENE_DATA__ = {blob};\n"
        f"{scene}\n"
        "</script>"
    )
    # </script> 가 문자열 안에 들어가면 파서가 일찍 닫으므로 방어한다
    inline = inline.replace("</script>", "<\\/script>", 1).replace("<\\/script>", "</script>", 1)
    out = page.replace(SRC_TAG, inline)

    dst = HERE / "standalone.html"
    dst.write_text(out, encoding="utf-8")
    print(f"저장 → {dst.relative_to(HERE.parents[1])}  {dst.stat().st_size:,} bytes")
    print("  Streamlit: components.html(open(...).read(), height=760, scrolling=True)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
