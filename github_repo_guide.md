# Consolidating `tiktok-skills` and `instagram-agent-skill` into a Unified Agent Architecture

## 1. Conceptual Hierarchy (`Platform -> Skill -> Action`)

Keeping `tiktok-skills` and `instagram-agent-skill` in separate repositories forces LLM agents to load duplicate tool descriptions for video publishing and analytics, wasting context tokens and increasing tool-selection ambiguity. Merging them establishes a single **Social Media Layer**:

| Layer | Responsibility | Concrete Implementation |
| :--- | :--- | :--- |
| **Skill Facade** | Single entry point for LLM tool calls (`social_upload`, `get_video_stats`, `repurpose_across_platforms`) | `SocialAgentSkill` |
| **Abstraction Contract** | Enforces Dependency Inversion (DIP) and capability discovery | `SocialMediaPlatform` + `CapabilityDescriptor` |
| **Platform Adapters** | Encapsulates platform-specific algorithmic logic, rate limits, and wire DTOs | `TikTokAdapter` (Velocity/Hook) & `InstagramAdapter` (Grid/Saves) |
| **Cross-Cutting Decorators** | Handles token-bucket rate limiting and exponential backoff retries | `RetryDecorator` |

## 2. Blending Algorithmic Philosophies

| Dimension | TikTok (`TikTokAdapter`) | Instagram (`InstagramAdapter`) |
| :--- | :--- | :--- |
| **Primary Signal** | Watch Time > Completion Rate > Shares | Saves > Shares > Profile Visits > Likes |
| **Aspect Ratio** | `9:16` vertical (keep text inside central safe zone: avoid top 15% and bottom 25%) | `4:5` portrait for feed; `9:16` for Reels with grid-harmonized cover frame |
| **Sweet-Spot Duration** | 7 to 15 seconds for viral completion rate | 15 to 45 seconds for Reels; multi-slide carousels for saves |
| **Metadata Strategy** | 3 to 5 niche semantic tags + searchable keywords in caption | Pre-fold hook + Save CTA + 5 to 10 tags + descriptive `alt_text` |
| **Velocity Cadence** | Up to 3x/day; evaluate completion rate at 60 minutes | 1 to 2x/day; allow 72+ hours for search/hashtag long-tail indexing |

## 3. Repository Consolidation Lifecycle (6-Phase Migration)

1. **Phase 0: Audit & Inventory** - Run `pipdeptree --freeze` on both repositories to identify conflicting versions (`pydantic`, `requests`, `httpx`).
2. **Phase 1: Dependency Unification** - Standardize on `pydantic>=2.0` as the single source of truth for JSON Schema export (`model_json_schema()`).
3. **Phase 2: Namespace Resolution** - Preserve legacy imports temporarily under `consolidated/legacy_tiktok/` and `consolidated/legacy_instagram/` with deprecation shims while moving shared logic into `scripts/social_agent_skill.py`.
4. **Phase 3: Class Hierarchy Refactoring** - Replace standalone `tiktok_upload()` and `instagram_upload()` functions with `BasePlatformAdapter` subclasses.
5. **Phase 4: Backward Compatibility Layering** - Keep thin wrapper aliases for existing scripts during transition.
6. **Phase 5 & 6: Contract Testing & Cleanup** - Verify both adapters return identical `SkillEnvelope` structures and remove legacy shims.
