"""
ground_logger.py - save live Anand Vihar ground readings (optional stand-alone tool)

You do NOT have to run this if you run the server: the server saves readings
by itself while it is running.  Use this file when you want readings saved even
when the server is closed (e.g. leave it running in a second window overnight).

    python ground_logger.py --check    # one test fetch, shows the numbers (run this FIRST)
    python ground_logger.py --once     # save the newest reading and stop
    python ground_logger.py            # keep saving, every 20 minutes (Ctrl+C to stop)
    python ground_logger.py --status   # how many hours are saved so far

It needs your free WAQI token in the WAQI_TOKEN environment variable.
"""
import argparse
import os
import sys
import time

import ground_feed as gf


def need_token():
    token = os.environ.get("WAQI_TOKEN", "")
    if not gf.has_token(token):
        print("WAQI_TOKEN is not set.")
        print("  1. Get a free token:  https://aqicn.org/data-platform/token/")
        print("  2. PowerShell:        $env:WAQI_TOKEN = \"paste-token-here\"")
        print("     Anaconda Prompt:   set WAQI_TOKEN=paste-token-here")
        sys.exit(1)
    return token


def do_check(token):
    data = gf.fetch_waqi(token)
    print("Station          :", (data.get("city") or {}).get("name"))
    print("Station time     :", (data.get("time") or {}).get("s"), (data.get("time") or {}).get("tz", ""))
    print("WAQI overall AQI :", data.get("aqi"), "(US scale - not the Indian AQI)")
    iaqi = data.get("iaqi") or {}
    for key in ("pm25", "pm10", "t", "h", "w"):
        print("  iaqi.%-5s    : %s" % (key, (iaqi.get(key) or {}).get("v")))
    print()
    print("Converted back to ug/m3 and then to the Indian CPCB AQI:")
    for scheme in ("epa2012", "epa2024"):
        rec = gf.parse_waqi(data, scheme)
        mark = "   <-- currently used" if scheme == gf.PM25_SCHEME else ""
        print("  PM2.5 table %-8s: PM2.5 = %6.1f ug/m3, PM10 = %6.1f ug/m3  ->  CPCB AQI %s%s"
              % (scheme, rec["pm25_ugm3"], rec["pm10_ugm3"], rec["AQI"], mark))
    print()
    print("HOW TO CHECK: open the Anand Vihar page on aqi.in (or CPCB's AQI site) for the SAME hour.")
    print("Pick the table whose PM2.5 ug/m3 matches that page, and the CPCB AQI should be within")
    print("about 10-15 points of the official Anand Vihar figure.")
    print("If the 'epa2024' line matches better, set  WAQI_PM25_SCHEME=epa2024  before starting.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        s = gf.status(os.environ.get("WAQI_TOKEN", ""))
        for k, v in s.items():
            print("%-18s %s" % (k, v))
        return
    token = need_token()
    if args.check:
        do_check(token)
        return
    while True:
        try:
            print(time.strftime("%H:%M:%S"), gf.poll_once(token))
        except Exception as exc:
            print(time.strftime("%H:%M:%S"), "poll failed:", exc)
        if args.once:
            return
        time.sleep(gf.POLL_SECONDS)


if __name__ == "__main__":
    main()