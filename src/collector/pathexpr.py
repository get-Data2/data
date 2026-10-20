"""설정 파일의 경로 표현식 해석기.

문법(contracts/config.md "경로 표현식"):
  a.b.c              중첩 키
  a[*].b             배열의 모든 원소 → 여러 값
  a[0], topic[-2]    배열 원소 / 토픽 조각(`topic`은 '/'로 나눈 토픽)
  values.*           사전의 각 키로 내려감. 내려간 키는 `$key`로 참조
  min(e), first(e)   여러 값 중 최소 / 첫 값
  split(e, "-", 0)   문자열 나누기
  "문자열", 3        상수
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

_STOP = set('.[](),"') | {" ", "\t"}


class PathExprError(ValueError):
    def __init__(self, src: str, pos: int, reason: str):
        super().__init__(f"{reason} (식 {src!r}, 위치 {pos})")
        self.src, self.pos, self.reason = src, pos, reason


# ---------- 노드 ----------

@dataclass(frozen=True)
class _Lit:
    value: Any

    def eval(self, ctx):
        return [self.value], False


@dataclass(frozen=True)
class _Seg:
    name: str              # 키 이름, '*', '$key'
    index: int | str | None  # None, 정수, '*'


@dataclass(frozen=True)
class _Path:
    segs: tuple[_Seg, ...]

    def eval(self, ctx):
        payload, topic = ctx
        multi = False
        first = self.segs[0]
        if first.name == "topic":
            items = [(topic.split("/"), None)]
        else:
            items = [(payload, None)]
        for i, seg in enumerate(self.segs):
            if not (i == 0 and seg.name == "topic"):
                items = _step_name(items, seg.name)
            if seg.index is not None:
                if seg.index == "*":
                    multi = True
                    items = [(e, k) for v, k in items if isinstance(v, list) for e in v]
                else:
                    out = []
                    for v, k in items:
                        if isinstance(v, list) and -len(v) <= seg.index < len(v):
                            out.append((v[seg.index], k))
                    items = out
        return [v for v, _ in items if v is not None], multi


def _step_name(items, name):
    out = []
    for v, k in items:
        if name == "*":
            if isinstance(v, dict):
                out.extend((cv, ck) for ck, cv in v.items())
        elif name == "$key":
            if k is not None:
                out.append((k, k))
        elif isinstance(v, dict) and name in v:
            out.append((v[name], k))
    return out


def _time_or_self(v):
    """ISO 시각 문자열은 시각으로 비교한다(소수 자릿수가 달라도 순서가 맞게)."""
    if isinstance(v, str):
        from .timeutil import parse_time
        t = parse_time(v)
        if t is not None:
            return (0, t)
        return (1, v)
    return (0, v) if isinstance(v, datetime) else (2, v)


@dataclass(frozen=True)
class _Func:
    name: str
    args: tuple

    def eval(self, ctx):
        vals, _ = self.args[0].eval(ctx)
        if self.name == "first":
            return vals[:1], False
        if self.name == "min":
            if not vals:
                return [], False
            try:
                return [min(vals, key=_time_or_self)], False
            except TypeError:
                return [], False
        if self.name == "split":
            sep = self.args[1].eval(ctx)[0][0]
            n = self.args[2].eval(ctx)[0][0]
            out = []
            for v in vals:
                parts = str(v).split(sep)
                if -len(parts) <= n < len(parts):
                    out.append(parts[n])
            return out, False
        raise AssertionError(self.name)


_FUNCS = {"min": 1, "first": 1, "split": 3}


# ---------- 구문 분석 ----------

class _Parser:
    def __init__(self, src: str):
        self.src, self.i = src, 0

    def err(self, reason):
        raise PathExprError(self.src, self.i, reason)

    def ws(self):
        while self.i < len(self.src) and self.src[self.i] in " \t":
            self.i += 1

    def peek(self):
        self.ws()
        return self.src[self.i] if self.i < len(self.src) else ""

    def expect(self, ch):
        if self.peek() != ch:
            self.err(f"'{ch}'가 필요함")
        self.i += 1

    def ident(self):
        self.ws()
        start = self.i
        while self.i < len(self.src) and self.src[self.i] not in _STOP:
            self.i += 1
        if start == self.i:
            self.err("이름이 필요함")
        return self.src[start:self.i]

    def expr(self):
        ch = self.peek()
        if ch == '"':
            self.i += 1
            end = self.src.find('"', self.i)
            if end < 0:
                self.err("닫는 따옴표 없음")
            s = self.src[self.i:end]
            self.i = end + 1
            return _Lit(s)
        if ch == "-" or ch.isdigit():
            start = self.i
            self.i += 1
            while self.i < len(self.src) and self.src[self.i].isdigit():
                self.i += 1
            try:
                return _Lit(int(self.src[start:self.i]))
            except ValueError:
                self.err("숫자 형식 오류")
        save = self.i
        name = self.ident()
        if self.peek() == "(":
            if name not in _FUNCS:
                self.i = save
                self.err(f"알 수 없는 함수 '{name}'")
            self.i += 1
            args = [self.expr()]
            while self.peek() == ",":
                self.i += 1
                args.append(self.expr())
            self.expect(")")
            if len(args) != _FUNCS[name]:
                self.err(f"함수 '{name}'의 인자는 {_FUNCS[name]}개")
            return _Func(name, tuple(args))
        self.i = save
        return self.path()

    def path(self):
        segs = []
        while True:
            name = self.ident()
            index = None
            if self.peek() == "[":
                self.i += 1
                if self.peek() == "*":
                    self.i += 1
                    index = "*"
                else:
                    self.ws()
                    start = self.i
                    if self.i < len(self.src) and self.src[self.i] == "-":
                        self.i += 1
                    while self.i < len(self.src) and self.src[self.i].isdigit():
                        self.i += 1
                    try:
                        index = int(self.src[start:self.i])
                    except ValueError:
                        self.err("배열 첨자는 정수 또는 *")
                self.expect("]")
            segs.append(_Seg(name, index))
            if self.peek() == ".":
                self.i += 1
                continue
            return _Path(tuple(segs))


class Expr:
    """컴파일된 경로 표현식."""

    def __init__(self, src: str):
        self.src = src
        p = _Parser(src)
        self._node = p.expr()
        if p.peek():
            p.err("식 뒤에 남은 문자")

    def eval(self, payload: Any, topic: str = "") -> Any:
        """여러 값 식([*])은 리스트, 그 밖에는 값 하나 또는 None을 돌려준다."""
        vals, multi = self._node.eval((payload, topic))
        if multi:
            return vals
        return vals[0] if vals else None

    def eval_items(self, payload: Any, topic: str = "") -> list[tuple[Any, Any]]:
        """`values.*`처럼 사전으로 내려가는 식에서 (값, 내려간 키) 쌍을 모두 돌려준다."""
        node = self._node
        if not isinstance(node, _Path):
            v = self.eval(payload, topic)
            return [] if v is None else [(v, None)]
        items = [(payload, None)]
        for seg in node.segs:
            items = _step_name(items, seg.name)
        return [(v, k) for v, k in items if v is not None]

    def __repr__(self):
        return f"Expr({self.src!r})"


def compile_expr(src: str) -> Expr:
    if not isinstance(src, str) or not src.strip():
        raise PathExprError(str(src), 0, "빈 식")
    return Expr(src)
