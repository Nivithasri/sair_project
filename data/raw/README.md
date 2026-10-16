# Raw data — provenance

Downloaded **2026-10-16**. Do not modify anything in this folder; derived files go to `data/processed/`.
All three sources are CC BY 4.0 (annotations). Twitter keeps ownership of tweet content.
Each original archive is kept unmodified next to the extracted subset, so the full data can be re-extracted.

| Folder | Role in project |
|---|---|
| `pheme_rumour_scheme/` | **Primary.** Tweets + human stance labels + who-follows-whom + retweets |
| `rumoureval2019/` | Cross-check of stance labels (same threads, SDQC labels) |
| `pheme_rnr/` | Fallback / robustness only: more threads, but no stance and no follow edges |

---

## 1. `pheme_rumour_scheme/` — PHEME rumour scheme dataset: journalism use case (v2)
- URL: https://figshare.com/articles/PHEME_rumour_scheme_dataset_journalism_use_case/2068650
  (file: https://ndownloader.figshare.com/files/4988998)
- DOI: 10.6084/m9.figshare.2068650.v2 · Licence: CC BY 4.0
- Archive: `pheme-rumour-scheme-dataset.tar.bz2`, 68,100,705 B, MD5 `533dc7205f7eaf67fd446aa2d863df51` (matches figshare)
- Cite: Zubiaga A, Liakata M, Procter R, Wong Sak Hoi G, Tolmie P (2016). *Analysing How People Orient to and
  Spread Rumours in Social Media by Looking at Conversational Threads.* PLoS ONE 11(3): e0150989.

Extracted (`pheme-rumour-scheme-dataset/`):
- `README`, `annotation-scheme.json`: the original documentation and label definitions
- `annotations/en-scheme-annotations.json`: one JSON per line (lines starting with `#` are section headers).
  Source tweets have `support` ∈ {supporting, denying, underspecified}. Replies have `responsetype-vs-source` and
  `responsetype-vs-previous` ∈ {agreed, disagreed, appeal-for-more-information, comment}, plus certainty and evidentiality.
- `threads/en/<event>/` for **all 8 English events** (charliehebdo 74, ebola-essien 2, ferguson 46, germanwings-crash 25,
  ottawashooting 58, prince-toronto 12, putinmissing 9, sydneysiege 71 threads). The other events were extracted on
  2026-10-16 to compare denier counts; the project uses charliehebdo (primary) + sydneysiege (secondary), and one ottawashooting story as a robustness set. Each `<thread_id>/` contains:
  - `source-tweets/<id>.json`, `reactions/<id>.json`: full Twitter API v1.1 tweet JSON
  - `retweets.json`: JSON lines, retweets of the source tweet (with user + timestamp)
  - `who-follows-whom.dat`: `follower_id<TAB>followee_id`, restricted to users within the thread
    (missing for 3 Germanwings threads)
  - `structure.json` (reply tree), `annotation.json` (thread-level rumour/veracity), `urls.dat`,
    `urls-content/`, `images/`, `images.dat`

**Not extracted (still inside the archive):** all German threads (`threads/de/`, including the German germanwings-crash);
`annotations/de-scheme-annotations.json`.

## 2. `rumoureval2019/` — RumourEval 2019 data (SemEval-2019 Task 7)
- URL: https://figshare.com/articles/RumourEval_2019_data/8845580 (file: https://ndownloader.figshare.com/files/16188500)
- DOI: 10.6084/m9.figshare.8845580.v1 · Licence: CC BY 4.0
- Archive: `rumoureval2019.tar.bz2`, 16,512,097 B, MD5 `85b37dc70106c2b9f584e63164fcbb92` (matches figshare)
- Cite: Gorrell G, Aker A, Bontcheva K, Kochkina E, Liakata M, Zubiaga A, Derczynski L (2019). *SemEval-2019 Task 7:
  RumourEval, Determining Rumour Veracity and Support for Rumours.* Proc. SemEval-2019, ACL, pp. 845–854.
- Note: Zenodo record 3269768 is **not** RumourEval. It is an augmented rumour/non-rumour set with no stance labels, so it was not used.

Extracted (`rumoureval2019/`): `LICENSE`, `README`, `final-eval-key.json`, `home_scorer_macro.py`, both zips (unmodified),
and from `rumoureval-2019-training-data.zip` only:
- `train-key.json`, `dev-key.json`: `subtaskaenglish` = tweet_id → {support, deny, query, comment}
  (**relative to the source tweet**, see DECISIONS D3). `subtaskbenglish` = veracity.
- `twitter-english/germanwings-crash/` (29 threads), `twitter-english/sydneysiege/` (74 threads). Each thread has
  `source-tweet/`, `replies/`, `structure.json`, `context/`, `urls.dat`. There are no follow edges and no retweets.

**Not extracted:** the other Twitter events, all Reddit data, and the test zip, which has no Germanwings or Sydney Siege threads.
The selected zip members were extracted with Python `zipfile`, because Git Bash `unzip` wildcards silently matched nothing.

## 3. `pheme_rnr/` — PHEME dataset of rumours and non-rumours (v1)
- URL: https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619
  (file: https://ndownloader.figshare.com/files/6453753)
- DOI: 10.6084/m9.figshare.4010619.v1 · Licence: CC BY 4.0
- Archive: `pheme-rnr-dataset.tar.bz2`, 26,735,006 B, MD5 `07f004eca867447d69c71e87be242d75` (figshare lists no MD5)
- Cite: Zubiaga A, Liakata M, Procter R (2016). *Learning Reporting Dynamics during Breaking News for Rumour Detection
  in Social Media.* arXiv:1610.07363 (the paper named in the dataset README). Dataset: Zubiaga, Wong Sak Hoi, Liakata, Procter (2016), figshare.

Extracted (`pheme-rnr-dataset/`): `README`, `germanwings-crash/{rumours (238), non-rumours (231)}`,
`sydneysiege/{rumours (522), non-rumours (699)}`. Each `<thread_id>/` has only `source-tweet/` and `reactions/`.

**Not extracted:** charliehebdo, ferguson, ottawashooting.

## 4. `nrc_emotion_lexicon/` — NRC Word-Emotion Association Lexicon (EmoLex) v0.92
- URL: https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip (official page: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- Downloaded 2026-10-16. Archive `NRC-Emotion-Lexicon.zip`, 25,878,449 B, MD5 `396b4706943a160894237913b025c171`
- **Licence / terms of use (from its README):** free for **non-commercial research and educational** use. You must cite
  the papers below and keep NRC's copyright notices. You may not rent, license or redistribute it to third parties.
  Commercial use needs a licence from NRC (saif.mohammad@nrc-cnrc.gc.ca). Because of this, the lexicon is git-ignored and not redistributed.
- Cite: Mohammad SM, Turney PD (2013). *Crowdsourcing a Word-Emotion Association Lexicon.* Computational Intelligence 29(3): 436–465.
  Mohammad SM, Turney PD (2010). *Emotions Evoked by Common Words and Phrases.* NAACL-HLT Workshop on Computational
  Approaches to Analysis and Generation of Emotion in Text.

Extracted: `NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` (word, emotion, 0/1; 14,154 words × 10
categories, of which we use the 8 emotions) and `NRC-Emotion-Lexicon/README.txt`.
**Not extracted:** the sense-level and other-language lexicons, and the PDFs.
