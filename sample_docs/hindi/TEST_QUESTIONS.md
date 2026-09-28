# Test questions for the Hindi sample documents

All organisations, schemes and phone numbers in these documents are **fictional**. The model
cannot know the answers from its training data, so a correct answer proves it came from
your knowledge base.

| File | Type | What it tests |
|---|---|---|
| `sunhari_fasal_yojana.docx` | Word | Headings, bullet lists and a **table** (crop-wise amounts) |
| `college_niyamawali.pdf` | 3-page text PDF | Hindi PDF extraction (conjuncts, matras) and **page-number citations** |
| `panchayat_suchna_scanned.pdf` | Scanned image PDF | **OCR** (needs `ENABLE_OCR=true` and Tesseract `hin`) |
| `dengue_bachav.txt` | Plain text | Simple Hindi text with some English words |
| `navrang_leave_policy.md` | Markdown | **Mixed Hindi + English** in the same document |

## 1. सुनहरी फसल सहायता योजना (DOCX)

| Question | Language | Expected answer |
|---|---|---|
| दलहन की फसल के लिए प्रति एकड़ कितनी सहायता मिलती है? | Hindi | 9,000 रुपये प्रति एकड़ (अधिकतम 45,000) |
| What is the last date to apply for the Kharif season? | English, cross-lingual | 31 October 2026 |
| Sunhari fasal yojana mein paisa kitni kiston mein milta hai? | Hinglish | 2 installments: 60% before sowing, 40% after harvest |
| क्या सरकारी कर्मचारी इस योजना के पात्र हैं? | Hindi | No, regular central or state government employees are not eligible |
| Which documents are required to apply? | English | Aadhaar, khasra-khatauni, bank passbook, residence certificate, 2 photos |
| Complaint ka nipटारा kitne din mein hota hai? | Hinglish + Devanagari | 15 working days; helpline 1800-000-4321 |

## 2. कॉलेज नियमावली (PDF, 3 pages)

| Question | Language | Expected answer (page) |
|---|---|---|
| न्यूनतम उपस्थिति कितनी होनी चाहिए? | Hindi | 75%, with up to 10% medical relaxation (p.1) |
| How much is the re-evaluation fee? | English, cross-lingual | ₹500 per subject, apply within 10 days (p.1) |
| Hostel mein raat ko kitne baje tak aana hai? | Hinglish | 9:30 pm (p.2) |
| पुस्तक देर से लौटाने पर कितना जुर्माना है? | Hindi | ₹5 per day per book (p.2) |
| What is the punishment for ragging? | English | Expulsion + fine up to ₹25,000 (p.3) |
| 85% marks par kitni fees maaf hoti hai? | Hinglish | 50% tuition fee waiver (p.3) |

## 3. पंचायत सूचना (scanned PDF, OCR)

| Question | Language | Expected answer |
|---|---|---|
| पानी की सप्लाई कब बंद रहेगी? | Hindi | 5 to 7 November 2026 |
| When and where will the water tanker come? | English, cross-lingual | 7 am near the Panchayat Bhavan, 5 pm near the primary school |
| Ek parivar ko kitna paani milega? | Hinglish | Up to 100 litres |
| इस सूचना पर किसके हस्ताक्षर हैं? | Hindi | Smt. Kamla Devi, Sarpanch |

## 4. डेंगू से बचाव (TXT)

| Question | Language | Expected answer |
|---|---|---|
| डेंगू का मच्छर कब काटता है? | Hindi | Mostly during the day (morning and evening) |
| Which painkillers should be avoided in dengue? | English, cross-lingual | Aspirin and ibuprofen; take only paracetamol, on a doctor's advice |
| Dengue ke danger signs kya hain? | Hinglish | Severe stomach pain, persistent vomiting, bleeding gums or nose, extreme weakness |

## 5. नवरंग लीव पॉलिसी (MD, mixed Hindi + English)

| Question | Language | Expected answer |
|---|---|---|
| साल में कितनी casual leave मिलती है? | Mixed | 12 days, max 3 at a time, no carry forward |
| How many earned leaves can be carried forward? | English | Accumulated EL capped at 30 days |
| Paternity leave kitne din ki hai? | Hinglish | 10 days, within 3 months of birth |
| क्या सोमवार को work from home ले सकते हैं? | Mixed | No, office attendance is mandatory on Mondays |

## 6. Tricky tests

| Question | What should happen |
|---|---|
| सुनहरी फसल योजना और कॉलेज छात्रवृत्ति में आय की सीमा क्या है? | Combines two documents: the scheme has no income limit (income-tax payers are excluded), while the scholarship needs family income below ₹2.5 lakh |
| भारत के प्रधानमंत्री कौन हैं? | **Not in the documents.** The assistant should say it could not find this in your documents, not answer from general knowledge |
| What is the hostel fee? | **Trap:** only the *mess* fee (₹3,200/month) is given. A good answer says the hostel fee is not mentioned |
| *(follow-up)* Ask "रैगिंग की शिकायत कहाँ करें?" and then "उनका फोन नंबर क्या है?" | Tests follow-up rewriting. Expected: 1800-000-7788 |

**Also test updating the knowledge base:** delete `navrang_leave_policy.md` in the UI and ask about casual leave again.
The answer should now say it can't find this. Re-upload the file and the answer comes back.
