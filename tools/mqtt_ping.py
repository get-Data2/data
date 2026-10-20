"""외부 패키지 없이 MQTT 브로커 로그인(CONNECT→CONNACK)만 확인한다."""
import os, socket, struct, sys

def env(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d

def s(b):
    return struct.pack("!H", len(b)) + b

e = env(sys.argv[1])
user, pw = e["MQTT_USER"].encode(), e["MQTT_PASS"].encode()
body = s(b"MQTT") + bytes([4, 0xC2]) + struct.pack("!H", 30) + s(b"jetson-probe") + s(user) + s(pw)
rl, n = b"", len(body)
while True:
    n, r = divmod(n, 128)
    rl += bytes([r | (0x80 if n else 0)])
    if not n:
        break
sock = socket.create_connection((e["MQTT_HOST"], int(e["MQTT_PORT"])), timeout=5)
sock.sendall(b"\x10" + rl + body)
ack = sock.recv(4)
sock.sendall(b"\xe0\x00")
codes = {0: "접속 성공", 1: "프로토콜 버전 거부", 2: "클라이언트 ID 거부", 3: "서버 사용 불가", 4: "ID/PW 틀림", 5: "권한 없음"}
print("MQTT:", codes.get(ack[3], f"알 수 없는 코드 {ack[3]}") if len(ack) == 4 and ack[0] == 0x20 else f"예상 밖 응답 {ack!r}")
