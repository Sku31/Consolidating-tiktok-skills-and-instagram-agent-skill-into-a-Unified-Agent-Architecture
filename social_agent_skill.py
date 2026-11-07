#!/usr/bin/env python3
"""
Unified Social Media Agent Skill (TikTok + Instagram)
=====================================================
Architecture: Platform -> Skill -> Action
Compatible with: OpenAI (GPT-4o), Anthropic (Claude 3.5), Meta (Llama 3),
                 LangChain, CrewAI, and Microsoft AutoGen.

Implements:
1. Dependency Inversion & Adapter Pattern (SocialMediaPlatform -> TikTokAdapter, InstagramAdapter)
2. Capability-Based Modeling (CapabilityDescriptor)
3. Decorator Pattern for Rate Limiting & Retry Backoff (RetryDecorator)
4. 3-Layer Normalization (Protocol DTO -> Domain Model -> Semantic Signal)
5. Single-Source-of-Truth Pydantic V2 Validation & Multi-LLM Schema Export
6. Structured Error Envelopes (SkillEnvelope) for Safe Agent Recovery
7. Observe-Think-Act (OTA) Autonomous Loop with Namespace Isolation
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, Field, ValidationError

def _validate_pydantic(model_cls, data: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)

def _export_json_schema(model_cls) -> dict:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()



# ============================================================================
# 1. CANONICAL ERROR TAXONOMY & AGENT ENVELOPE
# ============================================================================

class CanonicalErrorType(str, Enum):
    NONE = "NONE"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    AUTH_FAILURE = "AUTH_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    CONTENT_REJECTED = "CONTENT_REJECTED"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    NETWORK_ERROR = "NETWORK_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass
class SkillEnvelope:
    """Standardized agent observation output across all LLM frameworks."""
    success: bool
    platform: str
    action: str
    data: Optional[Dict[str, Any]] = None
    error_type: CanonicalErrorType = CanonicalErrorType.NONE
    message: str = "OK"
    retryable: bool = False
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["error_type"] = self.error_type.value
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# 2. CAPABILITY-BASED MODELING & DOMAIN OBJECTS (3-LAYER NORMALIZATION)
# ============================================================================

@dataclass(frozen=True)
class CapabilityDescriptor:
    platform_id: str
    supports_video_upload: bool
    supports_carousel: bool
    supports_alt_text: bool
    supports_trending_audio_attach: bool
    optimal_aspect_ratios: List[str]
    max_video_duration_sec: int
    sweet_spot_duration_sec: tuple[int, int]
    recommended_hashtag_range: tuple[int, int]
    max_daily_posts_recommended: int
    rate_limit_tokens_per_minute: int
    primary_algorithmic_signal: str


@dataclass
class NormalizedContentItem:
    """Layer 2: Canonical Domain Object for published media."""
    post_id: str
    platform: str
    media_url: str
    aspect_ratio: str
    duration_sec: Optional[int]
    caption: str
    hashtags: List[str]
    alt_text: Optional[str]
    published_at_epoch: float
    algorithmic_profile: str


@dataclass
class EngagementMetrics:
    """Layer 2 & 3: Normalized Domain Metrics + Unified Semantic Score."""
    post_id: str
    platform: str
    views: int
    likes: int
    comments: int
    shares: int
    saves: int
    watch_time_avg_sec: float
    completion_rate: float
    profile_visits: int
    primary_velocity_score: float  # Semantic normalization (0.0 - 1.0)
    recommended_next_action: str


# ============================================================================
# 3. SINGLE-SOURCE-OF-TRUTH PYDANTIC SCHEMAS (TOOL INPUTS)
# ============================================================================

class SocialUploadInput(BaseModel):
    """Unified input schema for publishing content across TikTok and Instagram."""
    platform: Literal["tiktok", "instagram"] = Field(
        ...,
        description="Target social media platform ('tiktok' for algorithmic reach, 'instagram' for curated grid/Reels)."
    )
    media_url: str = Field(
        ...,
        description="URL or local path to the video or image asset."
    )
    caption: str = Field(
        ...,
        min_length=1,
        max_length=2200,
        description="Post caption. For TikTok, embed searchable keywords; for Instagram, lead with a pre-fold hook and include a Save CTA."
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="List of hashtags (without '#'). TikTok optimal: 3-5 niche + 1-2 trending. Instagram optimal: 5-10 mixed tier."
    )
    aspect_ratio: Literal["9:16", "4:5", "1:1"] = Field(
        default="9:16",
        description="Media aspect ratio. Use '9:16' for TikTok and Instagram Reels; use '4:5' for Instagram portrait feed."
    )
    duration_sec: Optional[int] = Field(
        default=15,
        ge=1,
        le=600,
        description="Video duration in seconds. TikTok viral sweet spot is 7-15s."
    )
    alt_text: Optional[str] = Field(
        default=None,
        description="Accessibility and SEO alt-text description (strongly recommended for Instagram)."
    )
    trending_audio_id: Optional[str] = Field(
        default=None,
        description="Optional trending audio identifier for TikTok or Instagram Reels."
    )
    verify_grid_harmony: bool = Field(
        default=True,
        description="For Instagram, checks whether the asset palette aligns with the previous 9 grid posts."
    )


class VideoStatsInput(BaseModel):
    """Unified input schema for retrieving normalized engagement metrics."""
    platform: Literal["tiktok", "instagram"] = Field(
        ...,
        description="Target social media platform ('tiktok' or 'instagram')."
    )
    post_id: str = Field(
        ...,
        description="Unique platform post identifier to inspect."
    )
    window_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Time window in hours for velocity and retention analysis."
    )


class RepurposeContentInput(BaseModel):
    """Unified input schema for cross-platform asset adaptation."""
    master_media_url: str = Field(
        ...,
        description="High-resolution master video asset URL."
    )
    core_topic: str = Field(
        ...,
        description="Core topic or searchable keyword phrase (e.g., 'easy origami paper crane')."
    )
    niche_hashtags: List[str] = Field(
        ...,
        
        description="Base niche hashtags to tailor across platforms."
    )
    visual_description: str = Field(
        ...,
        description="Concise visual description used to auto-populate Instagram SEO alt-text."
    )


# ============================================================================
# 4. PLATFORM ABSTRACTION & TEMPLATE METHOD (SOLID / DIP)
# ============================================================================

class SocialMediaPlatform(ABC):
    """Contractual abstraction boundary for all social media adapters."""

    @abstractmethod
    def get_capabilities(self) -> CapabilityDescriptor:
        ...

    @abstractmethod
    def upload_content(self, payload: SocialUploadInput) -> NormalizedContentItem:
        ...

    @abstractmethod
    def get_video_stats(self, payload: VideoStatsInput) -> EngagementMetrics:
        ...

    @abstractmethod
    def normalize_error(self, raw_error: Exception) -> SkillEnvelope:
        ...


class BasePlatformAdapter(SocialMediaPlatform):
    """Template Method base class handling pre-checks, execution, and normalization."""

    def __init__(self) -> None:
        self._tokens = self.get_capabilities().rate_limit_tokens_per_minute

    def _pre_check_upload(self, payload: SocialUploadInput) -> None:
        caps = self.get_capabilities()
        if self._tokens <= 0:
            raise RuntimeError("RATE_LIMIT_EXCEEDED: Token bucket exhausted for current minute.")
        if payload.duration_sec and payload.duration_sec > caps.max_video_duration_sec:
            raise ValueError(
                f"UNSUPPORTED_CAPABILITY: Duration {payload.duration_sec}s exceeds "
                f"{caps.platform_id} max of {caps.max_video_duration_sec}s."
            )
        if payload.aspect_ratio not in caps.optimal_aspect_ratios:
            raise ValueError(
                f"UNSUPPORTED_CAPABILITY: Aspect ratio {payload.aspect_ratio} is non-optimal for "
                f"{caps.platform_id} (supported: {caps.optimal_aspect_ratios})."
            )
        self._tokens -= 1

    def upload_content(self, payload: SocialUploadInput) -> NormalizedContentItem:
        self._pre_check_upload(payload)
        raw_dto = self._call_upload_api(payload)
        return self._normalize_upload(raw_dto, payload)

    def get_video_stats(self, payload: VideoStatsInput) -> EngagementMetrics:
        if self._tokens <= 0:
            raise RuntimeError("RATE_LIMIT_EXCEEDED: Token bucket exhausted for current minute.")
        self._tokens -= 1
        raw_stats = self._call_stats_api(payload)
        return self._normalize_stats(raw_stats, payload)

    @abstractmethod
    def _call_upload_api(self, payload: SocialUploadInput) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _normalize_upload(self, raw_dto: Dict[str, Any], payload: SocialUploadInput) -> NormalizedContentItem:
        ...

    @abstractmethod
    def _call_stats_api(self, payload: VideoStatsInput) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _normalize_stats(self, raw_stats: Dict[str, Any], payload: VideoStatsInput) -> EngagementMetrics:
        ...

    def normalize_error(self, raw_error: Exception) -> SkillEnvelope:
        msg = str(raw_error)
        platform_id = self.get_capabilities().platform_id
        if "RATE_LIMIT_EXCEEDED" in msg:
            return SkillEnvelope(
                success=False,
                platform=platform_id,
                action="api_call",
                error_type=CanonicalErrorType.RATE_LIMITED,
                message=msg,
                retryable=True,
                suggestion="Wait 15 minutes or route through RetryDecorator with exponential backoff."
            )
        if "UNSUPPORTED_CAPABILITY" in msg:
            return SkillEnvelope(
                success=False,
                platform=platform_id,
                action="capability_check",
                error_type=CanonicalErrorType.UNSUPPORTED_CAPABILITY,
                message=msg,
                retryable=False,
                suggestion=f"Adjust media parameters to match {platform_id} CapabilityDescriptor."
            )
        return SkillEnvelope(
            success=False,
            platform=platform_id,
            action="execution",
            error_type=CanonicalErrorType.EXECUTION_ERROR,
            message=msg,
            retryable=True,
            suggestion="Inspect adapter credentials or network connectivity."
        )


# ============================================================================
# 5. CONCRETE PLATFORM ADAPTERS (TIKTOK & INSTAGRAM)
# ============================================================================

class TikTokAdapter(BasePlatformAdapter):
    """
    TikTok Adapter: Optimizes for watch time, completion rate, safe-zone overlays,
    and semantic keyword indexing ("Hook-and-Hold" velocity model).
    """

    def get_capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            platform_id="tiktok",
            supports_video_upload=True,
            supports_carousel=True,
            supports_alt_text=False,
            supports_trending_audio_attach=True,
            optimal_aspect_ratios=["9:16"],
            max_video_duration_sec=600,
            sweet_spot_duration_sec=(7, 15),
            recommended_hashtag_range=(3, 5),
            max_daily_posts_recommended=3,
            rate_limit_tokens_per_minute=10,
            primary_algorithmic_signal="completion_rate_and_watch_time"
        )

    def _call_upload_api(self, payload: SocialUploadInput) -> Dict[str, Any]:
        # Protocol layer simulation returning TikTok wire DTO
        return {
            "aweme_id": f"tt_{int(time.time() * 1000) % 1000000}",
            "create_time": time.time(),
            "safe_zone_verified": True,
            "audio_track": payload.trending_audio_id or "original_sound_tt",
            "indexed_keywords": payload.caption.split()[:6]
        }

    def _normalize_upload(self, raw_dto: Dict[str, Any], payload: SocialUploadInput) -> NormalizedContentItem:
        trimmed_tags = payload.hashtags[:5]  # Avoid over-tagging signal dilution on TikTok
        return NormalizedContentItem(
            post_id=raw_dto["aweme_id"],
            platform="tiktok",
            media_url=payload.media_url,
            aspect_ratio=payload.aspect_ratio,
            duration_sec=payload.duration_sec,
            caption=payload.caption,
            hashtags=trimmed_tags,
            alt_text=None,
            published_at_epoch=raw_dto["create_time"],
            algorithmic_profile="hook_and_hold_velocity (safe-zone centered, 3-5 niche/trending tags)"
        )

    def _call_stats_api(self, payload: VideoStatsInput) -> Dict[str, Any]:
        return {
            "play_count": 14200,
            "digg_count": 1150,
            "comment_count": 84,
            "share_count": 310,
            "collect_count": 420,
            "avg_watch_time_sec": 11.4,
            "completion_rate": 0.68,
            "profile_ clicks": 95
        }

    def _normalize_stats(self, raw_stats: Dict[str, Any], payload: VideoStatsInput) -> EngagementMetrics:
        completion = raw_stats["completion_rate"]
        # Semantic normalization: TikTok prioritizes completion rate > watch time > shares
        velocity_score = round(min(1.0, (completion * 0.65) + (raw_stats["share_count"] / 1000.0 * 0.35)), 3)
        next_action = (
            "Completion rate >= 60%: keep active and test follow-up part 2 clip."
            if completion >= 0.60
            else "Completion rate < 50% in early window: re-cut opening 2s hook and repost in 24h."
        )
        return EngagementMetrics(
            post_id=payload.post_id,
            platform="tiktok",
            views=raw_stats["play_count"],
            likes=raw_stats["digg_count"],
            comments=raw_stats["comment_count"],
            shares=raw_stats["share_count"],
            saves=raw_stats["collect_count"],
            watch_time_avg_sec=raw_stats["avg_watch_time_sec"],
            completion_rate=completion,
            profile_visits=raw_stats["profile_ clicks"],
            primary_velocity_score=velocity_score,
            recommended_next_action=next_action
        )


class InstagramAdapter(BasePlatformAdapter):
    """
    Instagram Adapter: Optimizes for grid harmony, Saves, Shares, Profile Visits,
    and vision SEO alt-text ("Credibility-and-Community" model).
    """

    def get_capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            platform_id="instagram",
            supports_video_upload=True,
            supports_carousel=True,
            supports_alt_text=True,
            supports_trending_audio_attach=True,
            optimal_aspect_ratios=["4:5", "9:16", "1:1"],
            max_video_duration_sec=90,
            sweet_spot_duration_sec=(15, 45),
            recommended_hashtag_range=(5, 10),
            max_daily_posts_recommended=2,
            rate_limit_tokens_per_minute=10,
            primary_algorithmic_signal="saves_shares_and_grid_cohesion"
        )

    def _call_upload_api(self, payload: SocialUploadInput) -> Dict[str, Any]:
        return {
            "ig_media_id": f"ig_{int(time.time() * 1000) % 1000000}",
            "timestamp": time.time(),
            "grid_harmony_passed": payload.verify_grid_harmony,
            "alt_text_indexed": bool(payload.alt_text)
        }

    def _normalize_upload(self, raw_dto: Dict[str, Any], payload: SocialUploadInput) -> NormalizedContentItem:
        caption = payload.caption
        if "save" not in caption.lower():
            caption = f"{caption} | Save this for later!"
        alt_text = payload.alt_text or f"Curated {payload.aspect_ratio} visual post: {payload.caption[:80]}"
        return NormalizedContentItem(
            post_id=raw_dto["ig_media_id"],
            platform="instagram",
            media_url=payload.media_url,
            aspect_ratio=payload.aspect_ratio,
            duration_sec=payload.duration_sec,
            caption=caption,
            hashtags=payload.hashtags[:10],
            alt_text=alt_text,
            published_at_epoch=raw_dto["timestamp"],
            algorithmic_profile="credibility_and_community (grid-harmonized, alt-text indexed, save-CTA)"
        )

    def _call_stats_api(self, payload: VideoStatsInput) -> Dict[str, Any]:
        return {
            "impressions": 6800,
            "like_count": 450,
            "comments_count": 62,
            "shares_count": 140,
            "saved_count": 390,
            "avg_reel_watch_sec": 19.5,
            "reel_completion_rate": 0.54,
            "profile_visits": 215
        }

    def _normalize_stats(self, raw_stats: Dict[str, Any], payload: VideoStatsInput) -> EngagementMetrics:
        saves_ratio = raw_stats["saved_count"] / max(1, raw_stats["impressions"])
        velocity_score = round(min(1.0, (saves_ratio * 8.0) + (raw_stats["profile_visits"] / 500.0 * 0.4)), 3)
        next_action = (
            "High save-to-impression ratio: keep in evergreen rotation and consider $10 boost after 72h."
            if saves_ratio >= 0.04
            else "Allow 72h long-tail indexing via hashtags/alt-text; engage with early comments."
        )
        return EngagementMetrics(
            post_id=payload.post_id,
            platform="instagram",
            views=raw_stats["impressions"],
            likes=raw_stats["like_count"],
            comments=raw_stats["comments_count"],
            shares=raw_stats["shares_count"],
            saves=raw_stats["saved_count"],
            watch_time_avg_sec=raw_stats["avg_reel_watch_sec"],
            completion_rate=raw_stats["reel_completion_rate"],
            profile_visits=raw_stats["profile_visits"],
            primary_velocity_score=velocity_score,
            recommended_next_action=next_action
        )


# ============================================================================
# 6. DECORATOR & REGISTRY PATTERNS
# ============================================================================

class RetryDecorator(SocialMediaPlatform):
    """Wraps any SocialMediaPlatform adapter with exponential backoff retry."""

    def __init__(self, delegate: SocialMediaPlatform, max_retries: int = 2, base_delay_sec: float = 0.05) -> None:
        self._delegate = delegate
        self._max_retries = max_retries
        self._base_delay_sec = base_delay_sec

    def get_capabilities(self) -> CapabilityDescriptor:
        return self._delegate.get_capabilities()

    def _execute_with_backoff(self, fn: Callable[[], Any]) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except ValueError:
                # Non-retryable capability/validation error
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self._max_retries:
                    time.sleep(self._base_delay_sec * (2 ** attempt))
        assert last_err is not None
        raise last_err

    def upload_content(self, payload: SocialUploadInput) -> NormalizedContentItem:
        return self._execute_with_backoff(lambda: self._delegate.upload_content(payload))

    def get_video_stats(self, payload: VideoStatsInput) -> EngagementMetrics:
        return self._execute_with_backoff(lambda: self._delegate.get_video_stats(payload))

    def normalize_error(self, raw_error: Exception) -> SkillEnvelope:
        return self._delegate.normalize_error(raw_error)


class PlatformRegistry:
    """Service locator mapping platform names to decorated adapter instances."""

    def __init__(self) -> None:
        self._adapters: Dict[str, SocialMediaPlatform] = {}

    def register(self, platform_name: str, adapter: SocialMediaPlatform) -> None:
        self._adapters[platform_name.lower()] = adapter

    def resolve(self, platform_name: str) -> SocialMediaPlatform:
        key = platform_name.lower()
        if key not in self._adapters:
            raise KeyError(f"Platform '{platform_name}' is not registered. Available: {list(self._adapters.keys())}")
        return self._adapters[key]

    def list_platforms(self) -> List[str]:
        return list(self._adapters.keys())


# ============================================================================
# 7. STATE MANAGEMENT & ISOLATED NAMESPACE MEMORY (OTA LOOP)
# ============================================================================

@dataclass
class OperationalPlatformState:
    platform_id: str
    posts_published_today: int = 0
    recent_post_ids: List[str] = field(default_factory=list)
    episodic_log: List[Dict[str, Any]] = field(default_factory=list)


class HybridAgentMemory:
    """
    3-Layer State & Memory Store preventing cross-platform contamination:
    - Global Strategic State: shared campaign theme & cross-platform insights
    - Platform Operational State: strictly isolated by platform_id namespace
    """

    def __init__(self, campaign_goal: str = "Grow organic reach and community saves") -> None:
        self.global_strategic_goal = campaign_goal
        self.semantic_insights: List[str] = []
        self._platform_states: Dict[str, OperationalPlatformState] = {
            "tiktok": OperationalPlatformState(platform_id="tiktok"),
            "instagram": OperationalPlatformState(platform_id="instagram"),
        }
        self.active_platform: Optional[str] = None

    def switch_context(self, target_platform: str) -> Dict[str, Any]:
        target = target_platform.lower()
        if target not in self._platform_states:
            self._platform_states[target] = OperationalPlatformState(platform_id=target)
        self.active_platform = target
        state = self._platform_states[target]
        return {
            "active_platform": target,
            "global_goal": self.global_strategic_goal,
            "isolated_namespace": f"{target}:*",
            "posts_published_today": state.posts_published_today,
            "recent_platform_history": state.episodic_log[-3:],
            "shared_strategic_insights": self.semantic_insights[-3:],
        }

    def record_episodic_event(self, platform: str, event_type: str, summary: str, strategic_takeaway: Optional[str] = None) -> None:
        state = self._platform_states[platform.lower()]
        if event_type == "social_upload":
            state.posts_published_today += 1
        state.episodic_log.append({
            "namespace_key": f"{platform.lower()}:{event_type}:{len(state.episodic_log) + 1}",
            "summary": summary,
            "timestamp": time.time()
        })
        if strategic_takeaway:
            # Abstracted insight stored in L3 Semantic Memory (safe for cross-platform sharing)
            self.semantic_insights.append(strategic_takeaway)


# ============================================================================
# 8. UNIFIED FACADE & MULTI-LLM TOOL EXPORT (`SocialAgentSkill`)
# ============================================================================

class SocialAgentSkill:
    """
    Unified Master Skill merging `tiktok-skills` and `instagram-agent-skill`.
    Exposes de-duplicated actions, OTA loop orchestration, and LLM tool schemas.
    """

    def __init__(self, registry: Optional[PlatformRegistry] = None, memory: Optional[HybridAgentMemory] = None) -> None:
        if registry is None:
            registry = PlatformRegistry()
            registry.register("tiktok", RetryDecorator(TikTokAdapter()))
            registry.register("instagram", RetryDecorator(InstagramAdapter()))
        self.registry = registry
        self.memory = memory or HybridAgentMemory()

    # --- Action 1: De-duplicated Upload (`social_upload`) ---
    def social_upload(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate input via Pydantic, switch platform context, and publish media."""
        try:
            payload = _validate_pydantic(SocialUploadInput, kwargs)
        except ValidationError as exc:
            return SkillEnvelope(
                success=False,
                platform=str(kwargs.get("platform", "unknown")),
                action="social_upload",
                error_type=CanonicalErrorType.SCHEMA_VIOLATION,
                message="Input validation failed against SocialUploadInput schema.",
                retryable=False,
                suggestion="Fix required fields or enum values and re-invoke.",
                details={"validation_errors": exc.errors()}
            ).to_dict()

        self.memory.switch_context(payload.platform)
        adapter = self.registry.resolve(payload.platform)

        try:
            item = adapter.upload_content(payload)
            self.memory.record_episodic_event(
                platform=payload.platform,
                event_type="social_upload",
                summary=f"Published {item.post_id} ({item.aspect_ratio})",
                strategic_takeaway=f"Active campaign asset deployed on {payload.platform}."
            )
            return SkillEnvelope(
                success=True,
                platform=payload.platform,
                action="social_upload",
                data=asdict(item),
                message=f"Successfully published post {item.post_id} on {payload.platform}."
            ).to_dict()
        except Exception as exc:
            return adapter.normalize_error(exc).to_dict()

    # --- Action 2: Standardized Telemetry (`get_video_stats`) ---
    def get_video_stats(self, **kwargs: Any) -> Dict[str, Any]:
        """Retrieve normalized engagement metrics and next-step algorithmic guidance."""
        try:
            payload = _validate_pydantic(VideoStatsInput, kwargs)
        except ValidationError as exc:
            return SkillEnvelope(
                success=False,
                platform=str(kwargs.get("platform", "unknown")),
                action="get_video_stats",
                error_type=CanonicalErrorType.SCHEMA_VIOLATION,
                message="Input validation failed against VideoStatsInput schema.",
                retryable=False,
                suggestion="Provide valid 'platform' ('tiktok'|'instagram') and 'post_id'.",
                details={"validation_errors": exc.errors()}
            ).to_dict()

        self.memory.switch_context(payload.platform)
        adapter = self.registry.resolve(payload.platform)

        try:
            metrics = adapter.get_video_stats(payload)
            self.memory.record_episodic_event(
                platform=payload.platform,
                event_type="get_video_stats",
                summary=f"Post {metrics.post_id} velocity={metrics.primary_velocity_score}",
                strategic_takeaway=f"Audience responds well to concise visual hooks (velocity={metrics.primary_velocity_score})."
            )
            return SkillEnvelope(
                success=True,
                platform=payload.platform,
                action="get_video_stats",
                data=asdict(metrics),
                message=f"Retrieved normalized metrics for {metrics.post_id} on {payload.platform}."
            ).to_dict()
        except Exception as exc:
            return adapter.normalize_error(exc).to_dict()

    # --- Action 3: Cross-Platform Ecosystem Repurposing ---
    def repurpose_across_platforms(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Takes one master asset and automatically adapts and publishes distinct variants
        for Instagram (polished 4:5 portrait, Save-CTA, alt-text, 6 tags) and
        TikTok (fast-cut 12s 9:16 vertical, keyword hook, 4 tags).
        """
        try:
            payload = _validate_pydantic(RepurposeContentInput, kwargs)
        except ValidationError as exc:
            return SkillEnvelope(
                success=False,
                platform="multi",
                action="repurpose_across_platforms",
                error_type=CanonicalErrorType.SCHEMA_VIOLATION,
                message="Invalid repurpose parameters.",
                retryable=False,
                details={"validation_errors": exc.errors()}
            ).to_dict()

        ig_res = self.social_upload(
            platform="instagram",
            media_url=f"{payload.master_media_url}?variant=ig_4x5_polished",
            caption=f"Master {payload.core_topic} step-by-step! Save this for your next session.",
            hashtags=(payload.niche_hashtags + ["creativestudio", "tutorial", "dailycraft", "aestheticgrid"])[:8],
            aspect_ratio="4:5",
            duration_sec=30,
            alt_text=payload.visual_description,
            verify_grid_harmony=True,
        )

        tt_res = self.social_upload(
            platform="tiktok",
            media_url=f"{payload.master_media_url}?variant=tt_9x16_12s_hook",
            caption=f"Wait till you see the end: {payload.core_topic} in 12 seconds",
            hashtags=(payload.niche_hashtags[:3] + ["fyp", "satisfying"]),
            aspect_ratio="9:16",
            duration_sec=12,
            trending_audio_id="trend_asmr_fold_01",
        )

        return SkillEnvelope(
            success=ig_res["success"] and tt_res["success"],
            platform="ecosystem",
            action="repurpose_across_platforms",
            data={"instagram_deployment": ig_res, "tiktok_deployment": tt_res},
            message="Executed de-duplicated cross-platform repurposing workflow."
        ).to_dict()

    # --- Multi-LLM Schema Exporters ---
    @staticmethod
    def _tool_specs() -> List[tuple[str, str, Type[BaseModel]]]:
        return [
            (
                "social_upload",
                "Upload and optimize a video or carousel post for TikTok (9:16 reach/velocity) or Instagram (4:5 or 9:16 curated grid/Reels).",
                SocialUploadInput,
            ),
            (
                "get_video_stats",
                "Fetch normalized engagement metrics (views, likes, saves, completion_rate, velocity_score) and recommended next actions.",
                VideoStatsInput,
            ),
            (
                "repurpose_across_platforms",
                "Adapt a single master asset into platform-optimized variants for both Instagram and TikTok in one workflow.",
                RepurposeContentInput,
            ),
        ]

    @classmethod
    def to_openai_tools(cls) -> List[Dict[str, Any]]:
        """Compatible with OpenAI GPT-4o, Llama 3 tool-calling, and Microsoft AutoGen."""
        tools = []
        for name, desc, model in cls._tool_specs():
            schema = _export_json_schema(model)
            schema["additionalProperties"] = False
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                }
            })
        return tools

    @classmethod
    def to_anthropic_tools(cls) -> List[Dict[str, Any]]:
        """Compatible with Anthropic Claude 3.5 Sonnet / Haiku tool use."""
        tools = []
        for name, desc, model in cls._tool_specs():
            tools.append({
                "name": name,
                "description": desc,
                "input_schema": _export_json_schema(model),
            })
        return tools


if __name__ == "__main__":
    skill = SocialAgentSkill()
    print("=== 1. EXPORTED OPENAI / LLAMA 3 / AUTOGEN TOOL SCHEMAS ===")
    print(json.dumps(skill.to_openai_tools(), indent=2))

    print("=== 2. OTA WORKFLOW DEMO (INSTAGRAM -> STATS -> TIKTOK REPURPOSE) ===")
    workflow_result = skill.repurpose_across_platforms(
        master_media_url="https://cdn.example.com/media/paper_crane_master.mp4",
        core_topic="how to fold an origami paper crane",
        niche_hashtags=["origami", "papercraft", "diy"],
        visual_description="Close-up of hands folding crisp white origami paper into a crane shape."
    )
    print(json.dumps(workflow_result, indent=2))

    ig_post_id = workflow_result["data"]["instagram_deployment"]["data"]["post_id"]
    stats_result = skill.get_video_stats(platform="instagram", post_id=ig_post_id, window_hours=24)
    print("=== 3. NORMALIZED OBSERVATION ENVELOPE ===")
    print(json.dumps(stats_result, indent=2))