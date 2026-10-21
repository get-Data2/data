"""보드 로컬 MQTT 브로커(wonil-broker): amqtt, 0.0.0.0:11883 (research "결정 기록").

- 구독: 익명 허용(읽기 전용)
- 발행: .env의 INFER_MQTT_USER/PASS 계정만, 토픽 team2/inference/v1/# 만
실행: `.venv/bin/python -m service.broker [--port N]` (deploy/wonil-broker.service)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass, field

from amqtt.plugins.base import BaseAuthPlugin

log = logging.getLogger("service.broker")


class ServiceAuthPlugin(BaseAuthPlugin):
    """사용자 이름이 없으면 익명(구독 전용 — 발행은 토픽 ACL이 막음). 이름이 있으면 발행 계정·비밀번호가 맞아야 한다."""

    async def authenticate(self, *, session) -> bool:
        user = getattr(session, "username", None)
        if not user:
            return True
        cfg = self.context.config
        want = cfg.password_sha256
        got = hashlib.sha256((getattr(session, "password", None) or "").encode()).hexdigest()
        return bool(cfg.user) and user == cfg.user and bool(want) and hmac.compare_digest(got, want)

    @dataclass
    class Config:
        user: str = ""
        password_sha256: str = ""


def broker_config(bind: str, port: int, user: str, password: str, publish_topics: list[str]) -> dict:
    return {
        "listeners": {"default": {"type": "tcp", "bind": f"{bind}:{port}"}},
        "plugins": {
            "service.broker.ServiceAuthPlugin": {"user": user or "",
                                                 "password_sha256": hashlib.sha256(password.encode()).hexdigest() if password else ""},
            # 구독 규칙(acl)을 비워 두면 모두 거부되므로 명시한다. 익명은 결과 토픽만 구독할 수 있다
            "amqtt.plugins.topic_checking.TopicAccessControlListPlugin": {
                "acl": {"anonymous": list(publish_topics), **({user: list(publish_topics)} if user else {})},
                "publish_acl": {user: list(publish_topics)} if user else {}},
        },
    }


async def serve(cfg_dict: dict, ready: asyncio.Event | None = None, stop: asyncio.Event | None = None):
    from amqtt.broker import Broker
    b = Broker(cfg_dict)
    await b.start()
    if ready:
        ready.set()
    try:
        await (stop.wait() if stop else asyncio.Event().wait())
    finally:
        await b.shutdown()


def main(argv=None) -> int:
    from dotenv import load_dotenv

    from .config import load_service_config
    ap = argparse.ArgumentParser(prog="service.broker")
    ap.add_argument("--config", default="config/service.yaml")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)
    cfg = load_service_config(args.config)
    load_dotenv(".env", override=False)
    m, b = cfg["mqtt"], cfg["broker"]
    user, pw = os.environ.get(m["user_env"], ""), os.environ.get(m["pass_env"], "")
    if not user or not pw:
        log.warning("발행 계정(.env %s/%s)이 없음 — 이 브로커에는 아무도 발행할 수 없다", m["user_env"], m["pass_env"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(serve(broker_config(b["bind"], args.port or b["port"], user, pw, b["publish_topics"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
