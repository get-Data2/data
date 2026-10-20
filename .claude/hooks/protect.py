"""PreToolUse 훅: README.md 수정과 .env 내용 노출을 막는다.

종료 코드 2 + stderr 메시지를 돌려주면 Claude Code가 해당 도구 호출을 차단한다.
"""
import json
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README = os.path.join(PROJECT, "README.md")
ENV = os.path.join(PROJECT, ".env")

# README에 쓰기를 일으키는 셸 패턴 (읽기 명령은 통과)
README_WRITE = re.compile(
    r"(>\s*\S*README\.md|sed\s+-i[^|;&]*README\.md|\b(rm|mv|truncate|tee|chmod)\b[^|;&]*README\.md"
    r"|\bcp\b[^|;&]*\sREADME\.md\s*($|[|;&])|git\s+(checkout|restore|rm)\b[^|;&]*README\.md)",
    re.IGNORECASE,
)
# .env 내용을 화면에 내보내는 셸 패턴 (source/. 로 읽어 쓰는 것은 통과)
ENV_PRINT = re.compile(r"\b(cat|less|more|head|tail|bat|nl|strings|xxd|od|grep)\b[^|;&]*(^|[\s/])\.env\b")


def deny(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def main():
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {}) or {}

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = os.path.realpath(inp.get("file_path") or inp.get("notebook_path") or "")
        if path == os.path.realpath(README):
            deny("README.md는 읽기 전용입니다(과제 안내서). 산출물은 별도 파일에 작성하세요.")
        if path == os.path.realpath(ENV) and tool != "Write":
            deny(".env는 사용자가 직접 편집합니다.")

    if tool == "Read":
        if os.path.realpath(inp.get("file_path", "")) == os.path.realpath(ENV):
            deny(".env에는 비밀값이 있어 읽어서 출력하지 않습니다. 코드에서 dotenv로 로드하세요.")

    if tool == "Bash":
        cmd = inp.get("command", "")
        if README_WRITE.search(cmd):
            deny("README.md를 변경하는 셸 명령은 차단됩니다.")
        if ENV_PRINT.search(cmd):
            deny(".env 내용을 출력하는 명령은 차단됩니다. 키 이름만 필요하면 `cut -d= -f1 .env`를 쓰세요.")

    sys.exit(0)


if __name__ == "__main__":
    main()
