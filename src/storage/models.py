from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnomalyEvent(Base):
    """ORM model for a detected anomaly event."""

    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    # Relative path to the event directory under events_dir
    event_dir = Column(String(512), nullable=False, unique=True)
    # NULL until the embedding is stored and indexed in FAISS
    faiss_index_id = Column(Integer, nullable=True)
    audio_path = Column(String(512), nullable=True)
    frame_path = Column(String(512), nullable=True)
    embedding_path = Column(String(512), nullable=True)
    # Bounding boxes from vision module, serialized as JSON array
    source_region_json = Column(Text, nullable=True)
    # Extra metadata: window_index, feature norms, etc.
    extra_json = Column(Text, nullable=True)

    # -- Multimodal scores (v0.3) ------------------------------------------
    # All nullable for backward compatibility with events written by older
    # versions; populated as each modality/fusion stage comes online.
    audio_score = Column(Float, nullable=True)
    video_score = Column(Float, nullable=True)
    combined_score = Column(Float, nullable=True)
    fast_audio_score = Column(Float, nullable=True)
    slow_audio_score = Column(Float, nullable=True)
    fast_video_score = Column(Float, nullable=True)
    slow_video_score = Column(Float, nullable=True)
    # Top contributing features (z-score explainability), JSON arrays
    top_audio_features = Column(Text, nullable=True)
    top_video_features = Column(Text, nullable=True)
    # "audio-driven" | "video-driven" | "multimodal"
    dominant_modality = Column(String(32), nullable=True)
