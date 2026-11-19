"""Name test: can Lucy learn her name in talk?

From a womb tissue, teach 4 lines ("who are you ? *CHI: I am Lucy ."), then ask "who are you ?"
N times (default 20) at temperature 0.6, and count the replies that contain "Lucy". Runs
`lucy talk` with framing (Kael = *MOT:, Lucy = *CHI:). The womb is never written: the talk
save goes to a temp file. Extra arguments go to `lucy talk` (e.g. --kael-weight 50).

With --ask "what is that ?" the asked question is a control: Lucy should not answer "Lucy" to
everything, only to "who are you ?".

usage: python3 scripts/name_test.py <lucy binary> <womb.bin> [--asks N] [--ask Q] [lucy talk args...]
"""
import os, subprocess, sys, tempfile

TEACH = "who are you ? *CHI: I am Lucy ."
ASK = "who are you ?"


def replies(out: str) -> list[str]:
    # Every turn prints "> " + " *CHI: " + reply on one line.
    return [l[len(">  *CHI: "):] for l in out.splitlines() if l.startswith(">  *CHI: ")]


def main() -> None:
    lucy, womb, *rest = sys.argv[1:]
    asks, ask = 20, ASK
    for flag in ("--asks", "--ask"):
        if flag in rest:
            i = rest.index(flag)
            if flag == "--asks":
                asks = int(rest[i + 1])
            else:
                ask = rest[i + 1]
            del rest[i : i + 2]
    script = "\n".join([TEACH] * 4 + [ask] * asks + ["/quit"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        cmd = [lucy, "talk", "--resume", womb, "--save", os.path.join(d, "t.bin"),
               "--temperature", "0.6", "--idle", "0", *rest]
        out = subprocess.run(cmd, input=script.encode(), capture_output=True, check=True).stdout
    rs = replies(out.decode("utf-8", "replace"))
    assert len(rs) == 4 + asks, f"expected {4 + asks} replies, got {len(rs)}"
    asked = rs[4:]
    hits = sum("Lucy" in r for r in asked)
    print(f"{os.path.basename(womb)} [{ask}] {' '.join(rest) or '(defaults)'}: {hits}/{asks} replies contain Lucy")
    for r in asked[:3]:
        print("   ", r)


if __name__ == "__main__":
    main()
