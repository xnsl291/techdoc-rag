"""LS ELECTRIC 다운로드 센터에서 국문 인버터 매뉴얼을 수집한다.

다운로드 센터는 로그인 없이 열려 있고 목록 API도 인증이 필요 없다.
응답에 파일명, 크기, 직접 다운로드 URL이 들어 있어 Manifest 필드 대부분을 여기서 얻는다.

수집한 PDF는 제3자 저작물이므로 저장소에 커밋하지 않는다.
Manifest에 출처 URL이 남으므로 언제든 다시 받을 수 있다.

사용법:
    python scripts/collect_manuals.py --list-only
    python scripts/collect_manuals.py --output-directory data/documents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

DOWNLOAD_CENTER_API = "https://ssq.ls-electric.com/api/guest/zdata/ssqdoc/dlCenter"
SOURCE_PAGE = "https://ssq.ls-electric.com/kr/ko/dlcenter"

# 저압 드라이브 제품군. UC-2에서 모델 간 사양을 비교하려면 같은 카테고리로 묶여 있어야 한다.
SEARCH_KEYWORDS = ("G100", "S100", "H100", "M100", "iS7", "iG5A", "iV5", "S300")

# 서버에 부담을 주지 않도록 요청 간격을 둔다. robots.txt는 전체 허용이지만
# 공개 API를 빠르게 두드릴 이유가 없다.
REQUEST_INTERVAL_SECONDS = 1.0

USER_AGENT = "techdoc-rag manual collector (research use; contact via GitHub xnsl291/techdoc-rag)"

# 본체 매뉴얼만 남긴다. 통신 옵션과 부가장치 설명서는 제품 사양을 담지 않아 UC-1 대상이 아니다.
OPTION_MANUAL_PATTERN = re.compile(
    r"Profibus|CANopen|RAPIEnet|EtherCAT|EtherNet|Profinet|DeviceNet|Modbus"
    r"|옵션|통신|Extension IO|리모트|제동|필터|CAD",
    re.IGNORECASE,
)
KOREAN_FILE_PATTERN = re.compile(r"_KR|_Kor|Korean|국문|한글", re.IGNORECASE)

# 같은 제품에 완전본과 간편본이 함께 올라와 있다. UC-1은 트러블슈팅과 파라미터 표가
# 필요하므로 완전본을 쓴다. 간편본은 설치와 기본 운전만 담고 있어 답할 수 있는 질문이 적다.
SIMPLE_MANUAL_PATTERN = re.compile(r"간편본|Simple|Quick", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ManualCandidate:
    """다운로드 센터의 문서 한 건."""

    document_id: str
    source_document_id: int
    title: str
    file_name: str
    file_size_megabytes: float
    updated_at: str
    download_url: str
    tags: tuple[str, ...]

    @property
    def is_full_manual(self) -> bool:
        return not SIMPLE_MANUAL_PATTERN.search(f"{self.title} {self.file_name}")

    @property
    def preference_rank(self) -> tuple[bool, str]:
        """같은 제품에서 무엇을 고를지 정하는 순위. 완전본이 우선이고 그다음이 최신순."""
        return (self.is_full_manual, self.updated_at)

    @property
    def model_key(self) -> str:
        """같은 제품의 여러 버전을 묶기 위한 키.

        제목 앞부분의 모델명으로 판단한다. 정확한 모델명은 본문을 봐야 알 수 있으므로
        여기서는 중복 제거 용도로만 쓰고, Manifest의 모델명은 사람이 확인해 채운다.
        """
        for keyword in SEARCH_KEYWORDS:
            if keyword.lower() in self.title.lower():
                return keyword
        return self.title


def encode_url_path(url: str) -> str:
    """URL 경로를 percent-encoding 한다.

    API가 돌려주는 URL은 두 형태가 섞여 있다. 파일명이 한글이면 인코딩되지 않은 상태로,
    공백이 있으면 이미 `%20`으로 인코딩된 상태로 온다.
    전자를 그대로 넘기면 urllib이 요청 라인을 ascii로 인코딩하다 실패하고,
    후자를 그냥 인코딩하면 `%25`로 이중 인코딩되어 404가 난다.
    한 번 풀었다가 다시 인코딩해 두 경우를 모두 맞춘다.
    """
    parts = urllib.parse.urlsplit(url)
    normalized_path = urllib.parse.quote(urllib.parse.unquote(parts.path))
    return urllib.parse.urlunsplit(parts._replace(path=normalized_path))


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def search_manuals(keyword: str) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "docTypes": "MANUAL",
            "keyword": keyword,
            "langs": "ko-KR",
            "order": "UPDATE",
            "pageNo": 1,
            "pageSize": 50,
            "period": "",
        }
    )
    return _get_json(f"{DOWNLOAD_CENTER_API}?{query}&targets=0&targets=10").get("list", [])


def to_candidate(document: dict) -> ManualCandidate | None:
    """API 응답 한 건을 후보로 바꾼다. 조건에 맞지 않으면 None을 준다.

    직접 다운로드 URL이 없는 레코드는 버린다. 같은 제품이 여러 번 등재되어 있고
    구버전 레코드일수록 URL이 비어 있다.
    """
    if OPTION_MANUAL_PATTERN.search(document.get("docTitle", "")):
        return None

    for file_info in document.get("files", []):
        file_name = file_info.get("fileName", "")
        download_url = file_info.get("blobUrlForLargeFile")
        if not download_url or not file_name.lower().endswith(".pdf"):
            continue
        if not KOREAN_FILE_PATTERN.search(file_name):
            continue
        return ManualCandidate(
            document_id=f"ls-{document['id']}",
            source_document_id=document["id"],
            title=document.get("docTitle", ""),
            file_name=file_name,
            file_size_megabytes=round(file_info.get("fileSize", 0) / 1024, 1),
            updated_at=document.get("updateTime", ""),
            download_url=download_url,
            tags=tuple(document.get("tags", [])),
        )
    return None


def collect_candidates() -> list[ManualCandidate]:
    """키워드별로 조회해 제품당 한 건만 남긴다. 완전본 우선, 같으면 최신순."""
    selected_by_model: dict[str, ManualCandidate] = {}
    for keyword in SEARCH_KEYWORDS:
        for document in search_manuals(keyword):
            candidate = to_candidate(document)
            if candidate is None:
                continue
            previous = selected_by_model.get(candidate.model_key)
            if previous is None or candidate.preference_rank > previous.preference_rank:
                selected_by_model[candidate.model_key] = candidate
        time.sleep(REQUEST_INTERVAL_SECONDS)
    return sorted(selected_by_model.values(), key=lambda item: item.model_key)


def download(candidate: ManualCandidate, output_directory: Path) -> tuple[Path, str]:
    """원본 PDF를 받아 저장하고 sha256을 돌려준다.

    이미 받아둔 파일이 있으면 다시 받지 않는다. 20MB짜리를 반복해서 받을 이유가 없다.
    """
    document_directory = output_directory / candidate.document_id
    document_directory.mkdir(parents=True, exist_ok=True)
    target_path = document_directory / "original.pdf"

    if not target_path.exists():
        request = urllib.request.Request(
            encode_url_path(candidate.download_url), headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            target_path.write_bytes(response.read())
        time.sleep(REQUEST_INTERVAL_SECONDS)

    return target_path, hashlib.sha256(target_path.read_bytes()).hexdigest()


def build_manifest_entry(candidate: ManualCandidate, sha256: str, byte_size: int) -> dict:
    """Manifest 한 줄.

    model_name_from_content와 page_count 등 본문을 열어야 알 수 있는 항목은 비워 둔다.
    파일명이 실제 모델과 다른 사례가 이미 확인되어, 자동으로 채우면 잘못된 값이 들어간다.
    """
    entry = asdict(candidate)
    entry.update(
        {
            "sha256": sha256,
            "file_size_bytes": byte_size,
            "downloaded_on": date.today().isoformat(),
            "source_page": SOURCE_PAGE,
            "model_name_from_content": None,
            "page_count": None,
            "pages_without_text_layer": None,
            "table_count": None,
        }
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="LS ELECTRIC 국문 인버터 매뉴얼 수집")
    parser.add_argument(
        "--output-directory", default="data/documents", help="원본 PDF 저장 위치"
    )
    parser.add_argument(
        "--manifest-path", default="data/manifest.json", help="Manifest 저장 위치"
    )
    parser.add_argument(
        "--list-only", action="store_true", help="다운로드 없이 후보 목록만 출력"
    )
    arguments = parser.parse_args()

    candidates = collect_candidates()
    print(f"후보 {len(candidates)}건\n")
    for candidate in candidates:
        print(
            f"  {candidate.model_key:8s} {candidate.file_name}"
            f"  ({candidate.file_size_megabytes}MB)"
        )

    if arguments.list_only:
        return

    output_directory = Path(arguments.output_directory)
    manifest = []
    failures: list[tuple[ManualCandidate, str]] = []
    print()
    for candidate in candidates:
        # 한 건이 실패해도 나머지는 계속 받는다. 링크가 끊긴 문서 하나 때문에
        # 20MB짜리를 다시 받게 만들 이유가 없다.
        try:
            path, sha256 = download(candidate, output_directory)
        except (urllib.error.URLError, OSError) as error:
            failures.append((candidate, str(error)))
            print(f"  실패 {candidate.document_id}  {candidate.file_name}  {error}")
            continue
        byte_size = path.stat().st_size
        manifest.append(build_manifest_entry(candidate, sha256, byte_size))
        print(f"  받음 {candidate.document_id}  {byte_size / 1024 / 1024:.1f}MB  {path}")

    manifest_path = Path(arguments.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest {len(manifest)}건: {manifest_path}")
    if failures:
        # 실패를 조용히 넘기면 Manifest가 완전한 것처럼 보인다.
        print(f"실패 {len(failures)}건 — 링크가 바뀌었을 수 있으므로 다운로드 센터에서 확인할 것")
        for candidate, reason in failures:
            print(f"  {candidate.document_id}  {candidate.file_name}  {reason}")
    print("본문을 열어야 알 수 있는 항목(실제 모델명, 페이지 수 등)은 비어 있음")


if __name__ == "__main__":
    main()
