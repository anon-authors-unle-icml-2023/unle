import socket


def should_use_float64() -> bool:
    hostname = socket.gethostname()
    return hostname in (
        "gpu-sr670-20",
        "gpu-sr670-21",
        "gpu-sr670-22",
    )
