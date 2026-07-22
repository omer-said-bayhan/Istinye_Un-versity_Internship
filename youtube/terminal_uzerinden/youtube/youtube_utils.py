# -*- coding: utf-8 -*-
"""
YouTube Data API v3 uzerinden bir videonun yorumlarini ceker.
API anahtari almak icin README.md'ye bak.
"""

import re

from googleapiclient.discovery import build


def extract_video_id(url_or_id: str) -> str:
    """Farkli YouTube link formatlarindan video ID'sini cikarir."""
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|&v=)([^#&?\/\s]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Video ID bulunamadi: {url_or_id}")


def fetch_comments(api_key: str, video_url_or_id: str, max_comments: int = 500) -> list:
    """Videonun en fazla max_comments adet ust-seviye yorumunu ceker.
    Yorumlar kapali/devre disi ise bos liste doner."""
    video_id = extract_video_id(video_url_or_id)
    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": snippet.get("authorDisplayName", ""),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception as e:
        # Yorumlar kapatilmis olabilir, video bulunamamis olabilir, quota bitmis olabilir vb.
        print(f"[UYARI] Yorumlar cekilirken hata olustu: {e}")

    return comments
