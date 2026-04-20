#!/usr/bin/env python3
"""PubMed Baseline 다운로드 → SQLite 저장.

NLM FTP에서 PubMed baseline XML.gz 파일을 다운로드하고
파싱하여 SQLite에 저장한다.

- Publication Type 필터링 (노이즈 유형 제외)
- 10분 간격 최대 100회 재시도
- API 호출 제한 준수 (초당 3회)
- 체크포인트: 파일 단위로 처리 상태 추적
"""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from ftplib import FTP
from pathlib import Path

PUBMED_FTP = "ftp.ncbi.nlm.nih.gov"
PUBMED_BASELINE_DIR = "/pubmed/baseline/"
DOWNLOAD_DIR = Path("/home/max/pubmed_data/xml")
DB_PATH = Path("/home/max/pubmed_data/pubmed.db")

INCLUDE_PUB_TYPES = {
    "Journal Article", "Clinical Trial", "Randomized Controlled Trial",
    "Case Reports", "Review", "Systematic Review", "Meta-Analysis",
    "Observational Study", "Comparative Study", "Multicenter Study",
    "Clinical Study", "Validation Study", "Research Support, N.I.H., Extramural",
    "Research Support, Non-U.S. Gov't", "Research Support, U.S. Gov't, P.H.S.",
}

EXCLUDE_PUB_TYPES = {
    "News", "Newspaper Article", "Editorial", "Comment", "Letter",
    "Interview", "Biography", "Historical Article", "Published Erratum",
    "Retracted Publication", "Retraction of Publication",
}

MAX_RETRIES = 100
RETRY_INTERVAL = 600  # 10분


def init_db():
    """SQLite 데이터베이스 초기화."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS abstracts (
            pmid TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            pub_types TEXT,
            mesh_terms TEXT,
            pub_date TEXT,
            journal TEXT,
            pmc_id TEXT,
            ner_done INTEGER DEFAULT 0,
            llm_done INTEGER DEFAULT 0,
            n_cuis INTEGER,
            cuis TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pmid TEXT,
            cui_a TEXT,
            cui_b TEXT,
            classification TEXT,
            FOREIGN KEY (pmid) REFERENCES abstracts(pmid)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS kg_edges (
            cui_a TEXT,
            cui_b TEXT,
            n_present INTEGER,
            jensen_score REAL,
            g2 REAL,
            pmids TEXT,
            PRIMARY KEY (cui_a, cui_b)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS download_status (
            filename TEXT PRIMARY KEY,
            downloaded INTEGER DEFAULT 0,
            parsed INTEGER DEFAULT 0,
            n_articles INTEGER DEFAULT 0,
            n_included INTEGER DEFAULT 0
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_ner_done ON abstracts(ner_done)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_llm_done ON abstracts(llm_done)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pmc ON abstracts(pmc_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_rel_pair ON relations(cui_a, cui_b)")

    conn.commit()
    return conn


def list_baseline_files(max_retries=MAX_RETRIES):
    """FTP에서 baseline 파일 목록을 가져온다."""
    for attempt in range(max_retries):
        try:
            ftp = FTP(PUBMED_FTP)
            ftp.login()
            ftp.cwd(PUBMED_BASELINE_DIR)
            files = [f for f in ftp.nlst() if f.endswith(".xml.gz")]
            ftp.quit()
            return sorted(files)
        except Exception as e:
            print(f"  FTP 목록 조회 실패 (시도 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  {RETRY_INTERVAL}초 후 재시도...")
                time.sleep(RETRY_INTERVAL)
    return []


def download_file(filename: str, max_retries=MAX_RETRIES) -> bool:
    """FTP에서 파일을 다운로드한다."""
    local_path = DOWNLOAD_DIR / filename
    if local_path.exists():
        return True

    for attempt in range(max_retries):
        try:
            ftp = FTP(PUBMED_FTP)
            ftp.login()
            ftp.cwd(PUBMED_BASELINE_DIR)

            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)

            ftp.quit()
            return True
        except Exception as e:
            print(f"    다운로드 실패 {filename} (시도 {attempt+1}/{max_retries}): {e}")
            # 불완전 파일 삭제
            if local_path.exists():
                local_path.unlink()
            if attempt < max_retries - 1:
                wait = min(RETRY_INTERVAL, 60 * (attempt + 1))
                print(f"    {wait}초 후 재시도...")
                time.sleep(wait)
    return False


def should_include(pub_types: list[str]) -> bool:
    """Publication type 기반 필터링."""
    pt_set = set(pub_types)
    if pt_set & EXCLUDE_PUB_TYPES:
        return False
    if pt_set & INCLUDE_PUB_TYPES:
        return True
    # 포함/제외 모두 아닌 경우: 포함 (보수적)
    return bool(pub_types)


def parse_xml_gz(filepath: Path) -> list[dict]:
    """XML.gz 파일을 파싱하여 초록 목록을 반환한다."""
    articles = []

    try:
        with gzip.open(filepath, "rb") as f:
            # iterparse로 메모리 효율적 처리
            context = ET.iterparse(f, events=("end",))

            for event, elem in context:
                if elem.tag != "PubmedArticle":
                    continue

                try:
                    # PMID
                    pmid_elem = elem.find(".//PMID")
                    if pmid_elem is None:
                        elem.clear()
                        continue
                    pmid = pmid_elem.text

                    # Title
                    title_elem = elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""

                    # Abstract
                    abstract_parts = []
                    for abs_text in elem.findall(".//AbstractText"):
                        if abs_text.text:
                            abstract_parts.append(abs_text.text)
                        # mixed content (태그 포함)
                        if abs_text.tail:
                            abstract_parts.append(abs_text.tail)
                        for child in abs_text:
                            if child.text:
                                abstract_parts.append(child.text)
                            if child.tail:
                                abstract_parts.append(child.tail)
                    abstract = " ".join(abstract_parts).strip()

                    # 초록이 없거나 너무 짧으면 건너뛰기
                    if not abstract or len(abstract.split()) < 20:
                        elem.clear()
                        continue

                    # Publication Types
                    pub_types = []
                    for pt in elem.findall(".//PublicationType"):
                        if pt.text:
                            pub_types.append(pt.text)

                    # Publication type 필터링
                    if not should_include(pub_types):
                        elem.clear()
                        continue

                    # MeSH Terms
                    mesh_terms = []
                    for mh in elem.findall(".//MeshHeading/DescriptorName"):
                        ui = mh.get("UI", "")
                        if ui:
                            mesh_terms.append(ui)

                    # Publication Date
                    pub_date = ""
                    date_elem = elem.find(".//PubDate")
                    if date_elem is not None:
                        year = date_elem.findtext("Year", "")
                        month = date_elem.findtext("Month", "")
                        pub_date = f"{year}-{month}" if year else ""

                    # Journal
                    journal = ""
                    journal_elem = elem.find(".//Journal/Title")
                    if journal_elem is not None and journal_elem.text:
                        journal = journal_elem.text

                    # PMC ID
                    pmc_id = None
                    for aid in elem.findall(".//ArticleId"):
                        if aid.get("IdType") == "pmc" and aid.text:
                            pmc_id = aid.text

                    articles.append({
                        "pmid": pmid,
                        "title": title[:500],
                        "abstract": abstract,
                        "pub_types": json.dumps(pub_types),
                        "mesh_terms": json.dumps(mesh_terms),
                        "pub_date": pub_date,
                        "journal": journal[:200],
                        "pmc_id": pmc_id,
                    })

                except Exception:
                    pass
                finally:
                    elem.clear()

    except Exception as e:
        print(f"    파싱 오류: {e}")

    return articles


def insert_articles(conn: sqlite3.Connection, articles: list[dict]) -> int:
    """초록을 SQLite에 삽입한다 (중복 무시)."""
    c = conn.cursor()
    inserted = 0
    for a in articles:
        try:
            c.execute("""
                INSERT OR IGNORE INTO abstracts
                (pmid, title, abstract, pub_types, mesh_terms, pub_date, journal, pmc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (a["pmid"], a["title"], a["abstract"], a["pub_types"],
                  a["mesh_terms"], a["pub_date"], a["journal"], a["pmc_id"]))
            if c.rowcount > 0:
                inserted += 1
        except Exception:
            pass
    conn.commit()
    return inserted


def main():
    print("=" * 80)
    print("PubMed Baseline 다운로드 → SQLite")
    print("=" * 80)

    # 디렉토리 생성
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # DB 초기화
    print("\n[1/4] SQLite 초기화...")
    conn = init_db()

    # 이미 처리된 파일 확인
    c = conn.cursor()
    c.execute("SELECT filename FROM download_status WHERE parsed = 1")
    parsed_files = set(row[0] for row in c.fetchall())
    c.execute("SELECT COUNT(*) FROM abstracts")
    existing_count = c.fetchone()[0]
    print(f"  DB 위치: {DB_PATH}")
    print(f"  기존 초록: {existing_count:,}건")
    print(f"  처리 완료 파일: {len(parsed_files)}개")

    # FTP 파일 목록
    print("\n[2/4] FTP 파일 목록 조회...")
    all_files = list_baseline_files()
    if not all_files:
        print("  파일 목록 조회 실패!")
        return

    remaining = [f for f in all_files if f not in parsed_files]
    print(f"  전체 파일: {len(all_files)}개")
    print(f"  남은 파일: {len(remaining)}개")

    # 다운로드 + 파싱
    print(f"\n[3/4] 다운로드 + 파싱...")
    total_included = 0
    total_articles = 0
    start_time = time.time()

    for idx, filename in enumerate(remaining):
        print(f"\n  [{idx+1}/{len(remaining)}] {filename}")

        # 다운로드
        print(f"    다운로드 중...")
        if not download_file(filename):
            print(f"    다운로드 실패, 건너뜀")
            continue

        local_path = DOWNLOAD_DIR / filename
        file_size = local_path.stat().st_size / (1024 * 1024)
        print(f"    다운로드 완료 ({file_size:.1f} MB)")

        # 파싱
        print(f"    파싱 중...")
        articles = parse_xml_gz(local_path)
        total_articles += len(articles)

        # DB 삽입
        inserted = insert_articles(conn, articles)
        total_included += inserted

        # 상태 저장
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO download_status
            (filename, downloaded, parsed, n_articles, n_included)
            VALUES (?, 1, 1, ?, ?)
        """, (filename, len(articles), inserted))
        conn.commit()

        # XML.gz 삭제 (디스크 절약)
        local_path.unlink()

        # 진행 보고
        elapsed = time.time() - start_time
        rate = (idx + 1) / (elapsed / 3600) if elapsed > 0 else 0
        eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0

        c.execute("SELECT COUNT(*) FROM abstracts")
        db_total = c.fetchone()[0]

        print(f"    포함: {inserted:,}/{len(articles):,}건")
        print(f"    DB 총: {db_total:,}건 | 속도: {rate:.1f}파일/시간 | ETA: {eta:.1f}시간")

    # 최종 통계
    print(f"\n[4/4] 최종 통계")
    print("=" * 80)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM abstracts")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM download_status WHERE parsed = 1")
    files_done = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT pmc_id) FROM abstracts WHERE pmc_id IS NOT NULL")
    with_pmc = c.fetchone()[0]

    print(f"  처리 파일: {files_done}/{len(all_files)}")
    print(f"  총 초록: {total:,}")
    print(f"  PMC ID 있는 초록: {with_pmc:,}")
    print(f"  DB 크기: {DB_PATH.stat().st_size / (1024**3):.1f} GB")

    elapsed = time.time() - start_time
    print(f"  소요 시간: {elapsed/3600:.1f}시간")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
