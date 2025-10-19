# extract_spikes.py
import json, re, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # no-op fallback

def parse_args():
    p = argparse.ArgumentParser(description="Extract hottest tweet windows for GG2013.")
    p.add_argument("--input", default="gg2013.json", type=Path, help="Input JSON array of tweets")
    p.add_argument("--outdir", default="spikes_out", type=Path, help="Output directory")
    p.add_argument("--top-k", type=int, default=30, help="Number of peak minutes to start from")
    p.add_argument("--window-min", type=int, default=7, help="Expand each peak ±N minutes")
    p.add_argument("--bin-sec", type=int, default=60, help="Histogram bin size in seconds (default: 60)")
    p.add_argument("--keywords", default=r"\b(wins?|awarded|goes\s+to|best)\b",
                   help="Regex used to prefilter relevant tweets (case-insensitive)")
    p.add_argument("--include-rt", action="store_true", help="Include retweets (by default RTs are skipped)")
    return p.parse_args()

def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array of tweet objects.")
    return data

def minute_bucket(ts_ms: int, bin_ms: int) -> int:
    return ts_ms // bin_ms

def relevant(tweet: Dict[str, Any], anchor_re: re.Pattern, include_rt: bool) -> bool:
    txt = tweet.get("text", "") or ""
    if not include_rt and txt.startswith("RT "):  # drop retweets by default
        return False
    return bool(anchor_re.search(txt))

def build_hist(tweets: Iterable[Dict[str, Any]], bin_ms: int,
               anchor_re: re.Pattern, include_rt: bool) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for t in tweets:
        if not relevant(t, anchor_re, include_rt):
            continue
        ts = t.get("timestamp_ms")
        if ts is None:
            continue
        try:
            m = minute_bucket(int(ts), bin_ms)
        except Exception:
            continue
        hist[m] = hist.get(m, 0) + 1
    return hist

def top_bins(hist: Dict[int, int], k: int) -> List[int]:
    return [m for m, _c in sorted(hist.items(), key=lambda x: x[1], reverse=True)[:k]]

def expand_and_merge(peaks: List[int], window_min: int) -> List[Tuple[int, int]]:
    spans = [(p - window_min, p + window_min) for p in peaks]
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(int(s), int(e)) for s, e in merged]

def select_tweets(tweets: Iterable[Dict[str, Any]], spans: List[Tuple[int, int]],
                  bin_ms: int, anchor_re: re.Pattern, include_rt: bool) -> List[List[Dict[str, Any]]]:
    buckets = [[] for _ in spans]
    for t in tweets:
        if not relevant(t, anchor_re, include_rt):
            continue
        ts = t.get("timestamp_ms")
        if ts is None:
            continue
        try:
            m = minute_bucket(int(ts), bin_ms)
        except Exception:
            continue
        for i, (s, e) in enumerate(spans):
            if s <= m <= e:
                buckets[i].append(t)
    return buckets

def dedup_and_sort(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in tweets:
        tid = t.get("id") or t.get("id_str") or (t.get("text"), t.get("timestamp_ms"))
        if tid in seen:
            continue
        seen.add(tid)
        out.append(t)
    # sort by timestamp if present
    out.sort(key=lambda x: int(x.get("timestamp_ms", 0)))
    return out

def save_outputs(spans: List[Tuple[int, int]], buckets: List[List[Dict[str, Any]]],
                 outdir: Path, bin_ms: int):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    combined = []
    for i, ((s, e), tweets) in enumerate(zip(spans, buckets), start=1):
        start_ms = s * bin_ms
        end_ms = (e + 1) * bin_ms - 1
        fn = outdir / f"spike_{i:02d}.json"
        fn.write_text(json.dumps(tweets, ensure_ascii=False), encoding="utf-8")
        summary.append({
            "index": i,
            "minute_start": s,
            "minute_end": e,
            "epoch_ms_start": int(start_ms),
            "epoch_ms_end": int(end_ms),
            "tweet_count": len(tweets),
            "file": fn.name
        })
        combined.extend(tweets)

    combined = dedup_and_sort(combined)
    (outdir / "combined_spikes.json").write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(buckets)} spike files, combined_spikes.json ({len(combined)} tweets), and summary.json in {outdir}/")

def main():
    args = parse_args()
    bin_ms = args.bin_sec * 1000
    anchor_re = re.compile(args.keywords, re.I)

    print("Loading tweets…")
    tweets = load_json_array(args.input)
    print(f"Loaded {len(tweets):,} tweets")

    print("Building histogram (per {}s)…".format(args.bin_sec))
    hist = build_hist(tweets, bin_ms, anchor_re, args.include_rt)
    print(f"Unique minute bins: {len(hist):,}")

    print(f"Selecting top {args.top_k} peak minutes…")
    peaks = top_bins(hist, args.top_k)
    print("Expanding peaks and merging overlaps…")
    spans = expand_and_merge(peaks, args.window_min)
    print(f"Final merged windows: {len(spans)}")
    for i, (s, e) in enumerate(spans, 1):
        print(f"  [{i:02d}] minute {s} → {e} (width {e - s + 1} min)")

    print("Collecting tweets within windows…")
    # tqdm just shows a single pass; selection itself is O(N*windows) but windows are few
    buckets = select_tweets(tweets, spans, bin_ms, anchor_re, args.include_rt)

    print("Writing outputs…")
    save_outputs(spans, buckets, args.outdir, bin_ms)

if __name__ == "__main__":
    main()
