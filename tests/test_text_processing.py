from bhasharag.chunker import chunk_pages
from bhasharag.loaders import Page, load_document
from bhasharag.retrieval import BM25Index, reciprocal_rank_fusion
from bhasharag.text_utils import detect_language, normalize_text, split_sentences, tokenize_for_bm25


def words(text):
    return len(text.split())


def test_detect_language():
    assert detect_language("What is the eligibility for this scheme?") == "en"
    assert detect_language("इस योजना के लिए पात्रता क्या है?") == "hi"
    assert detect_language("Is yojana ke liye eligibility kya hai?") == "hinglish"
    assert detect_language("PM Kisan योजना की eligibility क्या है?") == "mixed"


def test_split_sentences_handles_danda():
    parts = split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है। And an English one.")
    assert parts == ["यह पहला वाक्य है।", "यह दूसरा वाक्य है।", "And an English one."]


def test_bm25_tokens_keep_matras_and_drop_stopwords():
    toks = tokenize_for_bm25("किसानों को सम्मान निधि मिलती है।")
    assert "किसानों" in toks and "निधि" in toks  # vowel signs stay attached
    assert "को" not in toks and "है" not in toks
    assert "।" not in "".join(toks)


def test_normalize_text_nfc_and_whitespace():
    # "क़" can be one code point (U+0958) or KA + NUKTA; both must normalise identically.
    assert normalize_text("क़") == normalize_text("क़")
    assert normalize_text("  a​  \r\n\r\n\r\nx ") == "a\n\nx"
    assert normalize_text("infor-\nmation") == "information"


def test_chunker_respects_budget_and_overlaps():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))
    chunks = chunk_pages([Page(3, text)], words, max_tokens=30, overlap_tokens=8)
    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks)
    assert all(c.page == 3 for c in chunks)
    # the last sentence of chunk i is repeated at the start of chunk i+1
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_chunker_splits_giant_sentence():
    chunks = chunk_pages([Page(1, "word " * 500)], words, max_tokens=100, overlap_tokens=10)
    assert all(c.token_count <= 100 for c in chunks)
    assert sum(c.token_count for c in chunks) >= 500


def test_rrf_prefers_items_ranked_well_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0][0] == 2


def test_bm25_filter_and_empty():
    idx = BM25Index()
    idx.build([])
    assert idx.search("anything", 5) == []
    idx.build([(1, "apple banana"), (2, "banana cherry"), (3, "durian")])
    assert [cid for cid, _ in idx.search("banana", 5, allowed_ids={2})] == [2]


def test_load_txt_and_reject_unknown():
    doc = load_document("a.txt", "नमस्ते दुनिया\n\nHello".encode("utf-8"))
    assert doc.pages[0].text == "नमस्ते दुनिया\n\nHello"
    try:
        load_document("a.exe", b"x")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected UnsupportedFileType")


def test_clean_devanagari_extraction_collapses_doubled_signs():
    from bhasharag.loaders import clean_devanagari_extraction

    assert clean_devanagari_extraction("महाविद्याालय") == "महाविद्यालय"
    assert clean_devanagari_extraction("मूल्यांांकन") == "मूल्यांकन"
    assert clean_devanagari_extraction("रैगिंंग") == "रैगिंग"
    assert clean_devanagari_extraction("प्रत्येक विद्यालय") == "प्रत्येक विद्यालय"  # valid text untouched


def test_garbled_hindi_detection():
    from bhasharag.loaders import looks_garbled_hindi

    good = "प्रत्येक छात्र के लिए हर विषय में न्यूनतम 75 प्रतिशत उपस्थिति अनिवार्य है और नियम सख्त हैं"
    bad = "प्र ेक छात्र के िलए हर िवषय म  ूनतम 75 प्रितशत उप  ित अिनवाय  है और िनयम सख्त ह"
    assert not looks_garbled_hindi(good)
    assert looks_garbled_hindi(bad)
