from typing import List, Tuple, Iterable, Set, Dict
from pathlib import Path

def _build_matcher_ac(
    terms: List[str],
    whole_word: bool = True,
    case_sensitive: bool = False,
):
    import ahocorasick
    canon = (lambda s: s) if case_sensitive else (lambda s: s.lower())
    canon2idx: Dict[str, int] = {}
    unique_terms: List[str] = []
    canon_terms: List[str] = []
    for t in terms:
        c = canon(t)
        if c not in canon2idx:
            canon2idx[c] = len(unique_terms)
            unique_terms.append(t)
            canon_terms.append(c)
    A = ahocorasick.Automaton()
    for i, c in enumerate(canon_terms):
        A.add_word(c, i)
    A.make_automaton()
    lens = [len(c) for c in canon_terms]
    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def match_terms_in_text(text: str) -> Set[int]:
        if not text:
            return set()
        s = text if case_sensitive else text.lower()
        hit = set()
        for end, idx in A.iter(s):
            if whole_word:
                L = lens[idx]
                start = end - L + 1
                if start > 0 and _is_word_char(s[start - 1]):
                    continue
                if end + 1 < len(s) and _is_word_char(s[end + 1]):
                    continue
            hit.add(idx)
        return hit
    term2idx = {t: canon2idx[canon(t)] for t in terms}
    return match_terms_in_text, unique_terms, term2idx


def count_from_texts_ac(
    pairs: List[Tuple[str, str]],
    texts: Iterable[str],
    *,
    whole_word: bool = True,
    case_sensitive: bool = False,
    total_docs: int | None = None,
) -> List[Tuple[int, int, int]]:
    if not pairs:
        return []
    all_terms: List[str] = [t for xy in pairs for t in xy]
    match_terms_in_text, uniq_terms, term2idx = _build_matcher_ac(
        all_terms, whole_word=whole_word, case_sensitive=case_sensitive
    )
    pair_idx = [(term2idx[x], term2idx[y]) for (x, y) in pairs]
    from collections import defaultdict
    adj: Dict[int, list] = defaultdict(list)
    for k, (ix, iy) in enumerate(pair_idx):
        adj[ix].append((k, 1))
        adj[iy].append((k, 2))
    fx = [0] * len(uniq_terms)
    fxy = [0] * len(pairs)
    it = texts
    if total_docs is not None:
        try:
            from tqdm import tqdm
            it = tqdm(texts, total=total_docs, leave=False)
        except Exception:
            pass

    for doc in it:
        hits = match_terms_in_text(doc)
        if not hits:
            continue
        for i in hits:
            fx[i] += 1
        seen_mask: Dict[int, int] = {}
        adj_local = adj
        fxy_local = fxy
        for i in hits:
            for k, bit in adj_local.get(i, ()):
                prev = seen_mask.get(k, 0)
                new = prev | bit
                if new == 3 and prev != 3:
                    fxy_local[k] += 1
                if new != prev:
                    seen_mask[k] = new
    out: List[Tuple[int, int, int]] = []
    for k, (x, y) in enumerate(pairs):
        out.append((fx[term2idx[x]], fx[term2idx[y]], fxy[k]))
    return out


def iter_wiki_texts(
    wiki_path: str,
    *,
    text_column: str = "text",
    first_n: int = -1,
) -> Tuple[Iterable[str], int]:
    from datasets import load_from_disk, load_dataset

    p = Path(wiki_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {wiki_path}")

    has_disk_signature = (p / "dataset_info.json").exists() or (p / "state.json").exists()
    if has_disk_signature:
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys"):
            ds = ds["train"]
        total = len(ds)

        def _it():
            gc = text_column
            for ex in ds:
                yield ex.get(gc, "") or ""

        return _it(), total

    parquet_files = sorted(str(pp) for pp in p.rglob("*.parquet"))
    if parquet_files:
        if first_n > 0:
            parquet_files = parquet_files[:first_n]
        ds_cnt = load_dataset("parquet", data_files={"train": parquet_files}, split="train", streaming=False)
        total = len(ds_cnt)
        ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train", streaming=True)

        def _it():
            gc = text_column
            for ex in ds:
                yield ex.get(gc, "") or ""

        return _it(), total

    raise RuntimeError(f"No save_to_disk directory or parquet shards found in {wiki_path}.")


def count_pairs_in_wiki(
    pairs: List[Tuple[str, str]],
    wiki_path: str = "./wiki_data",
    *,
    text_column: str = "text",
    whole_word: bool = True,
    case_sensitive: bool = False,
    first_n: int = -1,
) -> List[Tuple[int, int, int]]:
    texts, total_docs = iter_wiki_texts(
        wiki_path, text_column=text_column, first_n=first_n
    )
    return count_from_texts_ac(
        pairs, texts,
        whole_word=whole_word,
        case_sensitive=case_sensitive,
        total_docs=total_docs,
    )


if __name__ == "__main__":
    pairs = [
        ("Apple", "Microsoft"),
        ("New York", "United States"),
        ("York", "New York"),
        ("banana", "apple"),
    ]
    res = count_pairs_in_wiki(pairs, wiki_path="./wiki_data", first_n=-1)
    print(res)
