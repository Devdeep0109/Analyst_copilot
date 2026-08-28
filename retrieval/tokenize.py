"""
retrieval/tokenize.py

Tokenization for financial filings. Shared by BM25 and by any lexical
component of the hybrid retriever, so both see identical terms.

WHY THIS ISN'T `text.lower().split()`
-------------------------------------
Default tokenizers are built for prose and quietly destroy the parts of a
filing that carry the answer:

  "$1,577"      -> a naive split gives "$1,577"; the gold answer says "1577".
                   They never match. Numbers ARE the answers here, so the
                   tokenizer emits both the raw form and a comma-stripped form.
  "(1,577)"     -> accounting negative. Parens must not glue to the digits.
  "10-K", "FY2018", "PP&E", "non-GAAP"
                -> splitting on punctuation shatters these into noise.
  "property, plant and equipment"
                -> the query says "capital expenditure". No lexical tokenizer
                   can bridge that; it is exactly the gap dense retrieval has
                   to close, and naming it here keeps us honest about what
                   BM25 can and cannot do.

STOPWORDS
---------
Only true function words are dropped. Finance words that look generic to a
general-purpose stoplist ("net", "total", "other", "current") are kept --
"total current liabilities" is three of them in a row and is a real line item.
"""
from __future__ import annotations

import re

# Deliberately minimal. Anything domain-bearing stays in.
STOPWORDS = frozenset("""
a an the and or of to in for on at by with from as is are was were be been being
this that these those it its
what which who whom whose how why when where
do does did done doing
i you he she we they them his her their our your my me us
have has had having will would shall should can could may might must
if then than so such but not no nor only own same too very just also
""".split())

# Question-framing noise that appears in nearly every FinanceBench question and
# therefore carries no signal ("Assume that you are a public equities analyst.
# Answer the following question by primarily using information shown in...").
# Left OUT of the default path -- see `drop_question_boilerplate`, which is a
# separate switch so we can measure whether it actually helps.
QUESTION_BOILERPLATE = frozenset("""
assume public equities analyst answer answering question questions response
relying details shown primarily using information provide give based
following company companies please respond
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&.\-]*", re.I)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_ZEROS_RE = re.compile(r"^(\d+)\.0+$")


def _normalize_numbers(text: str) -> str:
    """Strip thousands separators BEFORE tokenizing.

    Doing it first, rather than emitting both forms afterwards, matters: a
    token regex splits "1,577" into "1" and "577", and those fragments are
    pure noise -- "1" is one of the most common tokens in a filing, so it
    inflates the score of every page containing any small number. Normalizing
    up front yields exactly one clean token, "1577".
    """
    return _THOUSANDS_RE.sub("", text)


def tokenize(
    text: str,
    drop_stopwords: bool = True,
    drop_question_boilerplate: bool = False,
    expand_numbers: bool = True,
) -> list[str]:
    """Text -> BM25 terms."""
    text = _normalize_numbers(text.lower().replace("—", " ").replace("–", " "))
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.strip(".-")
        if not tok:
            continue
        if drop_stopwords and tok in STOPWORDS:
            continue
        if drop_question_boilerplate and tok in QUESTION_BOILERPLATE:
            continue
        out.append(tok)
        # "1577.00" and "1577" are the same figure. Filings write the bare
        # form in tables; gold answers write the padded form. Emit both.
        if expand_numbers:
            m = _TRAILING_ZEROS_RE.match(tok)
            if m:
                out.append(m.group(1))
    return out


def tokenize_query(text: str, drop_question_boilerplate: bool = True) -> list[str]:
    """Queries get boilerplate stripping by default; documents do not."""
    return tokenize(text, drop_question_boilerplate=drop_question_boilerplate)


if __name__ == "__main__":
    samples = [
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)",
        "Total current liabilities 7,244 7,687",
        "non-GAAP 10-K PP&E FY2018 $1,577.00",
    ]
    for s in samples:
        print(f"\n{s}")
        print(f"  doc  : {tokenize(s)}")
        print(f"  query: {tokenize_query(s)}")
