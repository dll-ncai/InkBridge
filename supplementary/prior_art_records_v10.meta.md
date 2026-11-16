# prior_art_records_v10 — provenance

- Purpose: novelty and related-work review of the InkBridge v10 manuscript (IJDAR / ICDAR 2027 journal track), reviewer lens 'novelty'.
- Query date: 2026-11-15 (all queries run the same day).
- Sources: OpenAlex works API (`/works?search=`, authenticated with the stored API key, no mailto); arXiv export API (`/api/query`, `search_query` with relevance sort; `id_list` for verification). Single-record verification by DOI from OpenAlex (`/works/doi:<doi>`).
- Caps: OpenAlex `per_page`=25 (first page only) per query; arXiv `max_results`=25 per query. No pagination. Absence claims derived from these searches are provisional.
- Filters: none at query time. For reading, titles were filtered locally by the regex `trajector|stroke|online|ink|order|temporal|urdu|arabic|script`; the CSV keeps all hits, unfiltered.
- Queries: 24 (14 OpenAlex, 10 arXiv); see prior_art_queries_v10.csv. One OpenAlex query ('handwriting trajectory recovery') hit an SSLError on its first attempt and succeeded on retry. The arXiv queries for MBR/self-consistency with diffusion returned mostly off-topic hits (arXiv `all:` matching).
- Records: 387 search hits (299 unique titles) plus 28 verification records (record_type = cited_in_ms_checked / proposed_missing / checked_not_proposed).
- Verification: each proposed missing reference was fetched as an individual record (title, authors, year, venue, DOI or arXiv id). The abstract was read where the record has one. LwF/EWC and UCL-MHTR have no abstract in OpenAlex, so they were checked on title, authors, venue and DOI only.
- Not verified: the characterisation of Nagamatsu et al. 2026 (no abstract in OpenAlex), of HATFormer [6] for horizontal flipping (abstract silent, full text not retrievable), and of ICDAR 2013 competition [13] as Arabic-script work (abstract silent).
